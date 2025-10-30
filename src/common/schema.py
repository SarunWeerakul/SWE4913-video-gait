COCO17=["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
def make(video,fps,w,h,frames): return {"video":video,"fps":fps,"width":w,"height":h,"frames":frames}
def frame(idx,ms,people): return {"frame_idx":idx,"ms":ms,"people":people}  # people=[{"kpts":[[x,y,conf]*17]}]
