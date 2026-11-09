from adafruit_servokit import ServoKit


class PanTilt:

    def __init__(self):
            """
            Initialization method
            """
    
            # setup servos
            self.kit = ServoKit(channels=16)
            self.panServo = self.kit.servo[0] # servo 0 is the one that pans
            self.tiltServo = self.kit.servo[1] # servo 1 handles tilt
    
            self.panServo.actuation_range = 270
            self.panServo.set_pulse_width_range(500, 2500)
            self.tiltServo.set_pulse_width_range(500, 2500)
            self.panServo.angle = 135
            self.tiltServo.angle = 90

            self.panServoCenter = 135
            self.tiltServoCenter = 90
    
            # min and max tilt angle to stop camera hitting stuff
            self.minTiltAngle = 30
            self.maxTiltAngle = 140

    def panTiltRelative(self, panAngle, tiltAngle):
        """
        method to adjust the pan and tilt system the camera is on

        Args:
            panAngle (int): the angle to look left or right relative to straight ahead
            tiltAngle (int): the angle to look up or down relative to straight ahead
        """

        actualPanAngle = self.panServoCenter + int(panAngle)
        actualTiltAngle = self.tiltServoCenter + int(tiltAngle)

        if actualPanAngle < 0: # if less than min angle
            actualPanAngle = 0
        elif actualPanAngle > 180: # if greater than max angle
            actualPanAngle = 180

        if actualTiltAngle < self.minTiltAngle: # if less than min angle 
            actualTiltAngle = self.minTiltAngle
        elif actualTiltAngle > self.maxTiltAngle: # if greater than max angle
            actualTiltAngle = self.maxTiltAngle

        self.panServo.angle = actualPanAngle
        self.tiltServo.angle = actualTiltAngle

    def panTiltAbsolute(self, panAngle, tiltAngle):
        """
        method to adjust the pan and tilt system the camera is on

        Args:
            panAngle (int): the angle to look left or right relative to straight ahead
            tiltAngle (int): the angle to look up or down relative to straight ahead
        """

        if panAngle < 0: # if less than min angle
            panAngle = 0
        elif panAngle > 180: # if greater than max angle
            panAngle = 180

        if tiltAngle < self.minTiltAngle: # if less than min angle 
            tiltAngle = self.minTiltAngle
        elif tiltAngle > self.maxTiltAngle: # if greater than max angle
            tiltAngle = self.maxTiltAngle

        self.panServo.angle = panAngle
        self.tiltServo.angle = tiltAngle

if __name__ == "__main__":
    panTiltSystem = PanTilt()

    while True:
        panAngle = int(input("Enter pan angle: "))
        tiltAngle = int(input("Enter tilt angle: "))
        panTiltSystem.panTiltRelative(panAngle, tiltAngle)