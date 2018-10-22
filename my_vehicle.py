class Vehicle:

    def __init__(self, name, ds, dv, lane_id, s_road):
        self.name = name           # Name in format TXX
        self.ds = ds               # Distance to ego vehicle [m] Sensor.Object.OB01.Obj.TXX.NearPnt.ds
        self.dv = dv               # Velocity difference to ego vehicle [m/s] Sensor.Object.OB01.Obj.TXX.NearPnt.dv
        self.lane_id = lane_id     # Traffic.TXX.Lane.Act.LaneId
        self.s_road = s_road       # Traffic.TXX.sRoad
