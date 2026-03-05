from ultralytics import YOLO


path=r"D:\QQData\实验数据_new\yolo11n20251022_19_29\weights\best.pt"
# path=r"C:\Users\AAAAA\Desktop\detect\yolo11_MAFPN_modifyX_C3k2_20260126_14_33\weights\best.pt"

test_image=r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Val\9e9546261431b15a7ca8961b970c1314.jpeg"
test_image=r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Val\17018bf07d4c620980d539650c6b228f_720.jpg"
test_image=r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Val\06db2f27cc705f85f61eafbead89079b_720.jpg"
test_image =r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Val\1e596cc117da0abff85da08dcf39e9fe_720.jpg"
test_image=r"C:\Users\AAAAA\Desktop\ultralytics\Custom_Val\f4ffd76e78336c3e3d79fe7a4582ccf2_720.jpg"
model = YOLO(path)
results = model.predict(
    source=test_image,  # 也可以是图片文件夹路径
    conf=0.5,  # 置信度阈值
    save=True,  # 保存预测结果图
    visualize=True  # 关键！开启特征可视化
)