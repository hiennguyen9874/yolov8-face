-   `python3 export.py --weights ./weights/yolov8n-face.pt --img-size 640 --batch-size 1 --dynamic-batch --end2end --topk-all 100 --iou-thres 0.5 --conf-thres 0.45 --device 0 --simplify`
-   `polygraphy surgeon sanitize ./weights/yolov8n-face.onnx   -o ./weights/yolov8n-face-converted.onnx --override-input-shapes "images:[-1,3,640,640]"`

## TensorRT

-   `python3 export.py --weights ./weights/yolov8n-face.pt --img-size 640 --batch-size 1 --dynamic-batch --end2end --topk-all 100 --iou-thres 0.5 --conf-thres 0.45 --device 0 --simplify --cleanup --trt`
- ```
/usr/src/tensorrt/bin/trtexec \
    --buildOnly \
    --preview=+fasterDynamicShapes0805 \
    --onnx=./weights/yolov8n-face.onnx \
    --saveEngine=./weights/yolov8n-face.engine \
    --memPoolSize=workspace:12288 \
    --fp16 \
    --shapes=images:4x3x640x640 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:4x3x640x640 \
    --maxShapes=images:4x3x640x640
```
