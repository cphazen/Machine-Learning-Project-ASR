import sys, os, math

pi = math.pi

class Phasor(object):

    """
    Basically a complex number only with
    functionality for getting and setting
    the angle, magnitude, imaginary, and
    real components
    """

    def __init__(self,
                 magnitude=None,
                 angle=None,
                 real_part=None,
                 imaginary_part=None,
                 approximate_pi_half=False):

        """
        Initialize the number. If magnitude
        and/or angle are given, the args
        for the real and imaginary parts of
        the number are added to the number
        after the magnitude and angle args
        have been used to create the number.

        NOTE: the angle is in radians.

        NOTE: the approximate_pi_half arg is
        a boolean arg that when set to true,
        if the angle is approximately equal
        to a multiple of pi/2 (accurate to
        within +/- 1.0e-06), the angle will
        be treated as the exact multiple.
        This means that if the angle is
        approximately (0*pi)/2, then the
        magnitude will be entirely real. If
        the angle is approximately pi/2, the
        magnitude will be entirely complex.
        If the angle is approximately pi,
        then the magnitude will be entirely
        real and multiplied by negative one.
        If the angle is approximately
        (3*pi)/2, then the magnitude will be
        entirely complex and multiplied by
        negative one.
        """

        self._phasor_angle = float(0) # ONLY set to anything non-zero if BOTH the imaginary and real components are zero (to preserve the angle)!
        self._phasor_real = float(0)
        self._phasor_imag = float(0)
        if magnitude: mag = float(magnitude)
        else: mag = float(0)
        if angle: ang = float(angle)
        else: ang = float(0)
        if real_part: real = float(real_part)
        else: real = float(0)
        if imaginary_part: imag = float(imaginary_part)
        else: imag = float(0)


        if mag and ang:
            self.set_phasor_magnitude(mag)
            if ang: self.set_phasor_angle(ang, approximate_pi_half=approximate_pi_half)

        if real:
            self._phasor_real = self._phasor_real + real

        if imag:
            self._phasor_imag = self._phasor_imag + imag

    def get_phasor_real(self):
        return self._phasor_real

    def set_phasor_real(self, value):
        self._phasor_real = float(value)
        if value: self._phasor_angle = float(0)

    def del_phasor_real(self):
        """
        sets the real component back to zero
        """
        self._phasor_real = float(0)

    real = property(fget=get_phasor_real,
                    fset=set_phasor_real,
                    fdel=del_phasor_real,
                    doc=None)

    def get_phasor_imag(self):
        return self._phasor_imag

    def set_phasor_imag(self, value):
        self._phasor_imag = float(value)
        if value: self._phasor_angle = float(0)

    def del_phasor_imag(self):
        """
        sets the imaginary component back to zero
        """
        self._phasor_imag = float(0)

    imag = property(fget=get_phasor_imag,
                    fset=set_phasor_imag,
                    fdel=del_phasor_imag,
                    doc=None)

    def get_phasor_angle(self):
        if self._phasor_real == float(0):
            if self._phasor_imag == float(0):
                return self._phasor_angle
            elif self._phasor_imag < float(0):
                return (-1)*((math.pi)/2)
            else:
                return math.pi/2
        return math.atan2(self._phasor_imag, self._phasor_real)

    def _get_component_from_angle(self, angle, real, approximate_pi_half=False):
        # arg 'real' is True if we want the real component returned,
        # and False if we want the imaginary component returned
        phasor_real = float(0)
        phasor_imag = float(0)
        if approximate_pi_half:
            notches = int(angle / (math.pi/2))
            if (angle % (math.pi/2)) >= (math.pi/4): notches = notches + 1
            notches = notches%4
            if notches is 1:
                phasor_imag = self.get_phasor_magnitude()
                phasor_real = float(0)
            elif notches is 2:
                phasor_real = (-1)*self.get_phasor_magnitude()
                phasor_imag = float(0)
            elif notches is 3:
                phasor_imag = (-1)*self.get_phasor_magnitude()
                phasor_real = float(0)
            else: # notches must be zero?
                phasor_real = self.get_phasor_magnitude()
                phasor_imag = float(0)
        else:
            mag = self.get_phasor_magnitude()
            phasor_real = mag * math.cos(angle)
            phasor_imag = mag * math.sin(angle)
        if real:
            return phasor_real
        else:
            return phasor_imag

    def set_phasor_angle(self, angle, approximate_pi_half=False):
        real = self._get_component_from_angle(angle, real=True, approximate_pi_half=approximate_pi_half)
        imag = self._get_component_from_angle(angle, real=False, approximate_pi_half=approximate_pi_half)
        self._phasor_real = real
        self._phasor_imag = imag
        if not real and not imag: self._phasor_angle = angle
        else: self._phasor_angle = float(0)

    def del_phasor_angle(self):
        """
        sets the real componen back to zero
        """
        self.set_phasor_angle(0, approximate_pi_half=True)

    angle = property(fget=get_phasor_angle,
                     fset=set_phasor_angle,
                     fdel=del_phasor_angle,
                     doc=None)

    def get_phasor_magnitude(self):
        return math.sqrt((self._phasor_real)*(self._phasor_real) + (self._phasor_imag)*(self._phasor_imag))

    def set_phasor_magnitude(self, magnitude):
        if not magnitude:
            self.del_phasor_magnitude()
        else:
            angle = self.get_phasor_angle()
            self._phasor_real = magnitude
            self._phasor_imag = float(0)
            self.set_phasor_angle(angle)
            self._phasor_angle = float(0)

    def del_phasor_magnitude(self):
        """
        sets the real componen back to zero
        """
        self._phasor_angle = self.get_phasor_angle()
        self._phasor_real = float(0)
        self._phasor_imag = float(0)

    magnitude = property(fget=get_phasor_magnitude,
                         fset=set_phasor_magnitude,
                         fdel=del_phasor_magnitude,
                         doc=None)

class Sinusoid(Phasor):
    """docstring for Sinusoid."""
    def __init__(self,
                 sampling_frequency=None,
                 normalized_frequency=None,
                 number_of_samples=None,
                 magnitude=None,
                 angle=None,
                 real_part=None,
                 imaginary_part=None,
                 approximate_pi_half=False):
        super(Sinusoid, self).__init__(magnitude=magnitude,
                                       angle=angle,
                                       real_part=real_part,
                                       imaginary_part=imaginary_part,
                                       approximate_pi_half=approximate_pi_half)

        if sampling_frequency is not None:
            self._sampling_frequency = float(sampling_frequency)
        else:
            self._sampling_frequency = None

        if normalized_frequency is not None:
            self._normalized_frequency = float(normalized_frequency)
        else:
            self._normalized_frequency = None

        if number_of_samples is not None:
            self._number_of_samples = int(number_of_samples)
        else:
            self._number_of_samples = None

    def get_value_at_time(self,
                          time=None,
                          discrete_time_index=None,
                          sampling_frequency=None, # in Hertz
                          normalized_frequency=None): # in radians

        # TODO: these checks should be performed in a separate method since
        # every time get_list_of_sinusoid_values is called, this method is
        # called by it x number of times where x is the number of time
        # samples in the list being generated

        if time is None and discrete_time_index is None:
            raise ImproperlyConfigured('Either time or discrete_time_index \
                                        must be provided!\n')

        if time is not None and discrete_time_index is not None:
            raise ImproperlyConfigured('Please provide either time or \
                                        discrete_time_index, but not \
                                        both\n')

        current_sampling_frequency = self._sampling_frequency
        current_normalized_frequency = self._normalized_frequency

        if sampling_frequency:
            current_sampling_frequency = float(sampling_frequency)
        if normalized_frequency:
            current_normalized_frequency = float(normalized_frequency)

        if current_sampling_frequency is None:
            raise ImproperlyConfigured('No sampling_frequency was provided \
                                        at initialization or when calling \
                                        this method!\n')
        if current_normalized_frequency is None:
            raise ImproperlyConfigured('No normalized_frequency was provided \
                                        at initialization or when calling \
                                        this method!\n')

        if time is not None:
            time = float(time)
            radian_angle = time
            radian_angle = radian_angle * current_normalized_frequency
            radian_angle = radian_angle * current_sampling_frequency
            radian_angle = radian_angle + self.angle
        else:
            #TODO: should we just change this to float()?
            discrete_time_index = int(discrete_time_index)
            radian_angle = discrete_time_index * current_normalized_frequency
            radian_angle = radian_angle + self.angle
        real_part = self._get_component_from_angle(radian_angle,
                                                   real=True)
        imag_part = self._get_component_from_angle(radian_angle,
                                                   real=False)
        return (real_part, imag_part)

    def get_list_of_sinusoid_values(self,
                                   sampling_frequency=None,
                                   normalized_frequency=None,
                                   number_of_samples=None):

        current_number_of_samples = self._number_of_samples
        if number_of_samples is not None:
            if not int(number_of_samples):
                raise Exception('number_of_samples cannot be zero!\n')
        if int(number_of_samples):
            current_number_of_samples = int(number_of_samples)

        list_of_complex_values = []
        for discrete_index in range(0, current_number_of_samples):
            result_tuple = self.get_value_at_time(
                               discrete_time_index=discrete_index,
                               sampling_frequency=sampling_frequency,
                               normalized_frequency=normalized_frequency)
            list_of_complex_values.append(complex(result_tuple[0],
                                                  result_tuple[1]))
        return list_of_complex_values

class ImproperlyConfigured(Exception):
    def __init__(self, message):
        super(ImproperlyConfigured, self).__init__(self, message)
