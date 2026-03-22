from PIL import Image

# 读入两张图
img_src = Image.open("clock.png")       # 想要被缩放的那张
img_ref = Image.open("orange.png")       # 作为尺寸参考的那张

# 获取参考图尺寸 (width, height)
target_size = img_ref.size

# 按参考图尺寸缩放
img_resized = img_src.resize(target_size, Image.Resampling.LANCZOS)

# 保存结果
img_resized.save("src_resized.png")