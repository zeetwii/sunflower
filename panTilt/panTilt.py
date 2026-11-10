try:
    from adafruit_servokit import ServoKit
except Exception as e:
    print(f"Error importing ServoKit: {e}. Running in simulation mode.")

class PanTilt:

    def __init__(self):
            """
            Initialization method
            """

            try:
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

                self.simulationMode = False
            except Exception as e:
                print("Error initializing PanTilt: ", e)
                print("Switching to simulation mode.")
                self.simulationMode = True

    def panTiltRelative(self, panAngle, tiltAngle):
        """
        method to adjust the pan and tilt system the camera is on

        Args:
            panAngle (int): the angle to look left or right relative to straight ahead
            tiltAngle (int): the angle to look up or down relative to straight ahead
        """

        if not self.simulationMode:
            actualPanAngle = self.panServoCenter + int(panAngle)
            actualTiltAngle = self.tiltServoCenter + int(tiltAngle)

            if actualPanAngle < 0: # if less than min angle
                actualPanAngle = 0
            elif actualPanAngle > 270: # if greater than max angle
                actualPanAngle = 270

            if actualTiltAngle < self.minTiltAngle: # if less than min angle 
                actualTiltAngle = self.minTiltAngle
            elif actualTiltAngle > self.maxTiltAngle: # if greater than max angle
                actualTiltAngle = self.maxTiltAngle

            self.panServo.angle = actualPanAngle
            self.tiltServo.angle = actualTiltAngle
        else: # if in simulation mode, just print the angles
            print(f"Simulated Pan Angle: {panAngle}, Simulated Tilt Angle: {tiltAngle}")


    def panTiltAbsolute(self, panAngle, tiltAngle):
        """
        method to adjust the pan and tilt system the camera is on

        Args:
            panAngle (int): the angle to look left or right relative to straight ahead
            tiltAngle (int): the angle to look up or down relative to straight ahead
        """

        if not self.simulationMode:
            if panAngle < 0: # if less than min angle
                panAngle = 0
            elif panAngle > 270: # if greater than max angle
                panAngle = 270

            if tiltAngle < self.minTiltAngle: # if less than min angle 
                tiltAngle = self.minTiltAngle
            elif tiltAngle > self.maxTiltAngle: # if greater than max angle
                tiltAngle = self.maxTiltAngle

            self.panServo.angle = panAngle
            self.tiltServo.angle = tiltAngle
        else: # if in simulation mode, just print the angles
            print(f"Simulated Pan Angle: {panAngle}, Simulated Tilt Angle: {tiltAngle}")

if __name__ == "__main__":
    panTiltSystem = PanTilt()

    while True:
        panAngle = int(input("Enter pan angle: "))
        tiltAngle = int(input("Enter tilt angle: "))
        panTiltSystem.panTiltRelative(panAngle, tiltAngle)