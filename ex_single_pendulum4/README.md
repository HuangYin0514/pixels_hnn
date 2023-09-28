# 从图像中学习动力学模型


## 方法

- 生成图像文件[generate_pixel_images.ipynb](/home/lbu/project/pixels_hnn/ex_single_pendulum4/analysis/generate_pixel_images.ipynb)

- 训练文件[shell](ex_single_pendulum4/run.sh)

    ```
    # 模拟图像噪声
    encoder_point = encoder_point2pixel(y, multiple)
    y = decoder_pixel2point(encoder_point[:-1], encoder_point[1:], multiple)
    yt = torch.stack([dynamics(t, yi).clone().detach().cpu() for yi in y.unsqueeze(0)])
    yt = yt.squeeze()
    yt = truncated_lambdas(yt, config.dof)
    t = t[:-1]
    ```
    

## 结果

- 结果文件[infer](ex_single_pendulum4/analysis/infer.ipynb)

## 总结

- 本质上是检验方法处理噪声的能力