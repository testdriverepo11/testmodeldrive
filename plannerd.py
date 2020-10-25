import gc
from cereal import car
from common.params import Params

from microserv.controls.lib.planner import Planner
from microserv.controls.lib.vehicle_model import VehicleModel
from microserv.controls.lib.pathplanner import PathPlanner
import cereal.messaging as messaging

def plannerd_thread(sm=None, pm=None):
  Params().get("CarParams", block=True);

  CP = car.CarParams.from_bytes(Params().get("CarParams", block=True))


  PL = Planner(CP)
  PP = PathPlanner(CP)
  VM = VehicleModel(CP)

  if sm is None:
    sm = messaging.SubMaster(['model','carState','controlsState','liveParameters'])

  if pm is None:
    pm = messaging.PubMaster(['plan', 'liveLongitudinalMpc', 'pathPlan', 'liveMpc'])

  #sm['liveParameters'].valid = True
  #sm['liveParameters'].sensorValid = True
  #sm['liveParameters'].steerRatio = CP.steerRatio
  #sm['liveParameters'].stiffnessFactor = 1.0

  while True:
    sm.update()
    if sm.updated['model']:
      print("updated")
      PP.update(sm, pm, CP, VM)
    #if sm.updated['radarState']:
      #PL.update(sm, pm, CP, VM, PP)


def main(sm=None, pm=None):
  plannerd_thread(sm, pm)

if __name__ == "__main__":
  main()
