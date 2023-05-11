# 跳一跳
## 相关技术
+ 人体检测与追踪；
+ 人体骨骼关键点检测；
+ 双视角做2DPose，进行匹配得到disparity，图像匹配微调，得到3dpose；
+ 直接左视图左2DPose，时序2DPose回归得到3dpose；

## 跳一跳游戏人体朝向、下蹲、上跳判断
```
# demo
python jump.py
```

## 模型依赖
[PersonDetAISports](http://model.dginternal.com/#/model2/detail/1214)

[HumanPose](http://192.168.2.132/#/model2/detail/1151)

[RayThreeD](http://model.dginternal.com/#/model2/detail/723)

## 坐标系转换及相机参数设置
### 双目立体匹配 相机坐标系转世界坐标系
+ lib/binoCamera.py    self.cam2GroundRT 设置为棋盘格外参标定获取到的RT
