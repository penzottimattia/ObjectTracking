#!/usr/bin/env python3
"""ROS2 multi-camera, multi-object batched 6-DoF tracker."""
import argparse, gc, logging, os, sys, time
from collections import deque
from pathlib import Path
import numpy as np
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(str(Path(__file__).resolve().parent.parent))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--objects",nargs="+"); p.add_argument("--object")
    p.add_argument("--cameras",nargs="+"); p.add_argument("--camera"); p.add_argument("--width",type=int,default=1280)
    p.add_argument("--height",type=int,default=720); p.add_argument("--fps",type=int,default=30)
    p.add_argument("--ros-rate","--hz",dest="ros_rate",type=float,default=0); p.add_argument("--est_refine_iter",type=int,default=2)
    p.add_argument("--track_refine_iter",type=int,default=5); p.add_argument("--confidence",type=float,default=.5)
    p.add_argument("--frame_id",default=None,help="single-camera override"); p.add_argument("--topic",default=None,help="legacy single-camera pose topic")
    p.add_argument("--topic-prefix",default="/object_pose"); p.add_argument("--image-topic",default="/camera/color/image_raw")
    p.add_argument("--reset-service",default="/reset_tracker"); p.add_argument("--track-frames",type=int,default=0)
    p.add_argument("--debug",type=int,default=1); p.add_argument("--no-vis",action="store_true"); p.add_argument("--mesh-origin",action="store_true")
    a=p.parse_args(); objects=a.objects or ([a.object] if a.object else None); cameras=a.cameras or ([a.camera] if a.camera else ["default"])
    if not objects: p.error("provide --objects or --object")
    if len(objects)>5 or len(objects)!=len(set(objects)): p.error("use 1-5 unique objects")

    import rclpy
    from geometry_msgs.msg import PoseStamped
    from sensor_msgs.msg import Image
    from std_srvs.srv import Trigger
    from cv_bridge import CvBridge
    from scipy.spatial.transform import Rotation as R
    from utils.batch_tracking import MultiCameraCapture, build_multi_camera_state, collect_tracking_units, register_all_cameras, track_batch
    from utils.tracking_utils import draw_multi_camera_vis, load_sam3, set_logging_format, set_seed
    set_logging_format(); set_seed(0); rclpy.init(); node=rclpy.create_node("object_tracker")
    display=None
    if not a.no_vis:
        from utils.display import TkDisplay
        display=TkDisplay(title="ObjectTracker")
    tracked,_,refiner,glctx=build_multi_camera_state(objects,cameras,a.debug,a.mesh_origin)
    _,sam=load_sam3(a.confidence); capture=MultiCameraCapture(cameras,a.width,a.height,a.fps)
    bridge=CvBridge(); pose_pubs={}; image_pubs={}
    for ci,cam in enumerate(cameras):
        for obj in tracked:
            if len(cameras)==1 and len(tracked)==1 and a.topic: topic=a.topic
            else: topic=f"{a.topic_prefix.rstrip('/')}/{cam}/{obj['name']}"
            pose_pubs[(ci,obj["name"])]=node.create_publisher(PoseStamped,topic,10)
        image_topic=a.image_topic if len(cameras)==1 else f"{a.image_topic.rstrip('/')}/{cam}"
        image_pubs[ci]=node.create_publisher(Image,image_topic,10)
    pending={"reset":False}; paused=False; remaining=a.track_frames if a.track_frames>0 else None
    def reset_cb(req,res):
        pending["reset"]=True; res.success=True; res.message="reset queued"; return res
    node.create_service(Trigger,a.reset_service,reset_cb)
    def reset():
        nonlocal paused,remaining
        for obj in tracked:
            for state in obj["camera_states"]: state.update(initialized=False,pose=None,pose_last=None)
        paused=False; remaining=a.track_frames if a.track_frames>0 else None; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
    def pose_msg(pose,frame_id):
        m=PoseStamped(); m.header.frame_id=frame_id; m.header.stamp=node.get_clock().now().to_msg()
        m.pose.position.x,m.pose.position.y,m.pose.position.z=map(float,pose[:3,3]); q=R.from_matrix(pose[:3,:3]).as_quat()
        m.pose.orientation.x,m.pose.orientation.y,m.pose.orientation.z,m.pose.orientation.w=map(float,q); return m
    fps_hist=deque(maxlen=30); next_pub=0.0
    try:
        while rclpy.ok():
            rclpy.spin_once(node,timeout_sec=0)
            if pending["reset"]: pending["reset"]=False; reset()
            if paused:
                if display: display.pump()
                time.sleep(.05); continue
            frames=capture.read()
            if frames is None: continue
            if not all(s["initialized"] for o in tracked for s in o["camera_states"]):
                register_all_cameras(tracked,frames,sam,a.est_refine_iter)
            else:
                t0=time.time(); units=collect_tracking_units(tracked,frames); track_batch(refiner,glctx,units,a.track_refine_iter)
                now=time.time(); publish=a.ros_rate<=0 or now>=next_pub
                if publish:
                    next_pub=now+(1/a.ros_rate if a.ros_rate>0 else 0)
                    for unit in units:
                        st=unit.obj["camera_states"][unit.camera_idx]
                        frame_id=a.frame_id if len(cameras)==1 and a.frame_id else unit.frame["frame_id"]
                        pose_pubs[(unit.camera_idx,unit.obj["name"])].publish(pose_msg(st["pose"],frame_id))
                    for ci,frame in enumerate(frames):
                        msg=bridge.cv2_to_imgmsg(frame["color_bgr"],encoding="bgr8"); msg.header.frame_id=frame["frame_id"]; msg.header.stamp=node.get_clock().now().to_msg(); image_pubs[ci].publish(msg)
                dt=time.time()-t0; fps_hist.append(1/dt if dt>1e-4 else 0)
                if remaining is not None:
                    remaining-=1
                    if remaining<=0: paused=True
            if display:
                key=display.show(draw_multi_camera_vis(frames,tracked,sum(fps_hist)/len(fps_hist) if fps_hist else 0))
                if key in ("q","Escape"): break
                if key=="r": reset()
    except KeyboardInterrupt: pass
    finally:
        capture.stop()
        if display: display.destroy()
        node.destroy_node(); rclpy.shutdown()
if __name__=="__main__": main()
