相机标定：输入相机IP地址
1，点击标定播放->先调整外圈，再调整内圈->确定
2，根据相机位置，选择车辆类型。如相机在人体前方选择面包车，如在右方选择卡车。
3，车道标定->调整上下左右放大缩小后，选择正确的车道号，填入车道序号中。对勾选中后，选中的车道会发生颜色变化。务必两次车道都选中变色后，输入幅度（建议大车7.5，小车15）->确定
4，人脸标定->调整大小位置同上，分别左右转头以及正视前方到指定位置，->选定于以确认，->计算可以显示本次标定的结果（左右对话框中前面的数值需要小于0.7）->确定
5，驾驶室标定->调整大小位置同上->确定
6，文件保存

文件下发：输入主板IP地址（用于程序升级时<需分开进行>，选择文件“mars_adas_algorithm”）
1，在主板开机后大约30秒，黄灯常亮时，点击连接。->文本框显示“连接成功” 如显示“连接失败”，再次点击连接尝试
2，选择之前保存的demarcate.cfg文件。
3，点击文件传输->文本框显示“传输中。。。完成” 如显示文件错误，需要重启软件和主板。重复文佳下发过程
4，标定结束->主板黄色灯熄灭，后会闪烁数次。之后进入正常工作状态。