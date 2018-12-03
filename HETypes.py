# HETypes.py
#
# Data types used by Hawkeye to describe/annotate juggling videos.
#
# Copyright 2018 Jack Boyce (jboyce@gmail.com)

from math import sqrt, sin, cos, acos
import statistics


class Balltag:
    """
    Represents a single observation of a ball's location.
    """
    def __init__(self, frame, x, y, radius, total_weight=1.0):
        self.frame = frame      # frame number in movie
        self.x = x              # pixel coordinates
        self.y = y              # OpenCV convention: y=0 at top of screen
        self.radius = radius    # radius in pixel units
        self.total_weight = total_weight
        self.arc = None         # finished Ballarc that tag is assigned to

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Ballarc:
    """
    Represents a parabolic trajectory through space. The 'tags' attribute is
    the Balltag objects assigned to the arc.
    """
    def __init__(self):
        # assigned in processing steps 2 and 3:
        self.f_peak = 0.0       # frame number (float) when arc peaks
        self.a = 0.0            # parameters defining parabolic path
        self.b = 0.0
        self.c = 0.0
        self.e = 0.0
        self.tags = set()       # set of attached Balltags
        self.id_ = None         # unique sequentially-assigned identifier

        # assigned in processing step 4:
        self.next = None
        self.prev = None
        self.hand_throw = None
        self.f_throw = None
        self.x_throw = None
        self.hand_catch = None
        self.f_catch = None
        self.x_catch = None
        self.height = None      # needed?
        self.x_origin = None    # needed?
        self.y_origin = None    # needed?
        self.run_id = None      # run number in video (starting at 1)
        self.throw_id = None    # throw number in run (starting at 1)
        self.error = None       # tuple of error values
        self.ideal = None       # 'ideal' (zero-error) version of arc
        self.close_arcs = None  # list of tuples describing closest other arcs

    def get_position(self, frame, notes):
        """
        Returns the position of the arc (in pixel coordinates) at a given
        frame number in the video.
        """
        frame -= self.f_peak
        x = self.a + self.b * frame
        y = self.c + self.e * frame * frame
        s, c = sin(notes['camera_tilt']), cos(notes['camera_tilt'])
        return (x * c + y * s, y * c - x * s)

    def get_distance_from_tag(self, tag, notes):
        x, y = self.get_position(tag.frame, notes)
        return sqrt((x - tag.x)**2 + (y - tag.y)**2)

    def get_distance_from_arc(self, arc2, frame, notes):
        x1, y1 = self.get_position(frame, notes)
        x2, y2 = arc2.get_position(frame, notes)
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def get_frame_range(self, notes):
        """
        Get the range of frames where this arc's location is in the viewable
        area of the movie.
        """
        width = notes['frame_width']
        height = notes['frame_height']
        frame_count = notes['frame_count']

        f_min_x = 0 if self.b == 0 else (self.f_peak - self.a / self.b)
        f_max_x = frame_count if self.b == 0 else (self.f_peak +
                                                   (width - self.a) / self.b)
        f_min_x, f_max_x = min(f_min_x, f_max_x), max(f_min_x, f_max_x)
        f_min_x = int(f_min_x + 1)
        f_max_x = int(f_max_x + 1)

        f_min_y = 0 if ((height - self.c) / self.e) < 0 else (
                self.f_peak - sqrt((height - self.c) / self.e))
        f_max_y = frame_count if self.e < 0 else (2 * self.f_peak - f_min_y)
        f_min_y, f_max_y = min(f_min_y, f_max_y), max(f_min_y, f_max_y)
        f_min_y = int(f_min_y + 1)
        f_max_y = int(f_max_y + 1)

        return max(0, f_min_x, f_min_y), min(frame_count, f_max_x, f_max_y)

    def get_tag_range(self):
        """
        Get the range of frames where this arc has tags assigned.
        """
        return min(t.frame for t in self.tags), max(t.frame for t in self.tags)

    def get_median_tag_radius(self):
        return statistics.median([tag.radius for tag in self.tags])

    def closest_approach_frame(self, arc2, notes):
        """
        Calculate the frame time when this arc makes its closest approach
        to another given arc.
        """
        try:
            f1_start, f1_end = self.f_throw, self.f_catch
        except AttributeError:
            f1_start, f1_end = self.get_frame_range(notes)

        try:
            f2_start, f2_end = arc2.f_throw, arc2.f_catch
        except AttributeError:
            f2_start, f2_end = arc2.get_frame_range(notes)

        if f1_start > f2_end or f2_start > f1_end:
            return None
        f_start, f_end = max(f1_start, f2_start), min(f1_end, f2_end)

        """
        Solve a third-degree polynomial equation for frame number f:
           A*f^3 + B*f^2 + C*f + D = 0
        This polynomial is from differentiating with respect to f the
        distance squared between the arc positions. To minimize roundoff
        problems we write the equation in terms of offset from the peak:
           f = frame - fp1
        """
        a1, b1, c1, e1, fp1 = self.a, self.b, self.c, self.e, self.f_peak
        a2, b2, c2, e2, fp2 = arc2.a, arc2.b, arc2.c, arc2.e, arc2.f_peak

        A = 2*(e1-e2)**2
        B = -6*e2*(e1-e2)*(fp1-fp2)
        C = (b1-b2)**2 + 2*(c1-c2)*(e1-e2) - 2*e2*(e1-3*e2)*(fp1-fp2)**2
        D = (a1-a2)*(b1-b2) + (fp1-fp2)*(b2*(b2-b1) +
                                         2*e2*(c2-c1 + e2*(fp1-fp2)**2))

        candidates = Ballarc.solve_cubic(A, B, C, D)
        candidates = [(f + fp1) for f in candidates
                      if f_start < (f + fp1) < f_end]
        candidates.extend([f_start, f_end])

        return min(candidates, key=lambda f:
                   self.get_distance_from_arc(arc2, f, notes))

    def solve_cubic(A, B, C, D):
        """
        Find real roots of cubic polynomial:
            A*x^3 + B*x^2 + C*x + D = 0
        """
        results = []

        if A == 0 and B == 0:
            # Linear case
            if C != 0:
                x = -D / C
                results.append(x)
        elif A == 0:
            # Quadratic case
            disc = C * C - 4 * B * D
            if disc >= 0:
                disc2 = sqrt(disc)
                x1 = (-C + disc2) / (2 * B)
                x2 = (-C - disc2) / (2 * B)
                results.append(x1)
                results.append(x2)
        else:
            # Cubic (general) case
            f = (3 * C / A - (B ** 2 / A ** 2)) / 3
            g = (2 * B**3 / A**3 - 9 * B * C / A**2 + 27 * D / A) / 27
            h = (g**2) / 4 + (f**3) / 27

            if f == 0 and g == 0 and h == 0:
                # Three real roots, all equal
                if (D / A) >= 0:
                    x = (D / A) ** (1 / 3.0) * -1
                else:
                    x = (-D / A) ** (1 / 3.0)
                results.append(x)
            elif h <= 0:
                # Three real roots
                i = sqrt(((g ** 2.0) / 4.0) - h)
                j = i ** (1 / 3.0)
                k = acos(-(g / (2 * i)))
                L = -j
                M = cos(k / 3.0)
                N = sqrt(3) * sin(k / 3.0)
                P = -B / (3.0 * A)

                x1 = 2 * j * cos(k / 3.0) - B / (3.0 * A)
                x2 = L * (M + N) + P
                x3 = L * (M - N) + P
                results.append(x1)
                results.append(x2)
                results.append(x3)
            else:
                # One real root and two complex roots
                R = -(g / 2.0) + sqrt(h)
                if R >= 0:
                    S = R ** (1 / 3.0)
                else:
                    S = (-R) ** (1 / 3.0) * -1
                T = -(g / 2.0) - sqrt(h)
                if T >= 0:
                    U = (T ** (1 / 3.0))
                else:
                    U = ((-T) ** (1 / 3.0)) * -1

                x1 = S + U - B / (3 * A)
                """
                x2 = -(S + U) / 2 - (B / (3.0 * a)) +
                     (S - U) * math.sqrt(3) * 0.5j
                x3 = -(S + U) / 2 - (b / (3.0 * a)) -
                     (S - U) * math.sqrt(3) * 0.5j
                """
                results.append(x1)

        for x in results:
            val = ((A * x + B) * x + C) * x + D
            if abs(val) > 1e-2:
                print(f'ERROR: f({A}, {B}, {C}, {D}; {x}) = {val}')

        return results
