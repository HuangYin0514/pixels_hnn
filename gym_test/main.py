import gym
import os
import imageio


# 指定保存图像的文件夹路径
result_folder = "result"

# 创建保存图像的文件夹（如果不存在）
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

episode_images = []  # 用于保存当前回合的图像

# 创建一个Gym环境
env = gym.make("Pendulum-v1", render_mode="rgb_array")
observation, info = env.reset(seed=42)

observation = env.reset()

for i in range(10):
    # 在每个时间步获取环境的图像
    img = env.render()
    episode_images.append(img)

    action = env.action_space.sample()  # 随机选择一个动作
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

    imageio.imsave(os.path.join(result_folder, f"frame_{i}.png"), img)


env.close()
