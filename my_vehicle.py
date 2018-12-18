

################################################################################
################################################################################

# This model script is used to save and access variables for each vehicle in each
# time step (scene)

################################################################################
################################################################################


class Vehicle:

    def __init__(self, name, lane_id, s_road, ds=None, dv=None, v=None, s_0=None, s_1=None, s_2=None, s_3=None):
        self.name = name           # Name in format TXX
        self.lane_id = lane_id     # Traffic.TXX.Lane.Act.LaneId or Car.Road.Lane.Act.LaneId
        self.s_road = s_road       # Traffic.TXX.sRoad or Car.Road.sRoad
        self.ds = ds               # Distance to ego vehicle [m] Sensor.Object.OB01.Obj.TXX.NearPnt.ds
        self.dv = dv               # Velocity difference to ego vehicle [m/s] Sensor.Object.OB01.Obj.TXX.NearPnt.dv
        self.v = v                 # Car.v
        self.s_0 = s_0             # v_ego * 3.6
        self.s_1 = s_1             # v_ego * 3.6 * 2/3
        self.s_2 = s_2             # v_ego * 3.6 * 1/2
        self.s_3 = s_3             # v_ego * 3.6 * 1/3
