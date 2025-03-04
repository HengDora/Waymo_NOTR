import os
import numpy as np
from lib.config import yacs
from datetime import datetime

def parse_cfg(cfg, args):
    # 检查任务名称是否为空，如果为空则抛出异常
    if len(cfg.task) == 0:
        raise ValueError('task must be specified')

    # 分配 GPU 资源
    if -1 not in cfg.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(gpu) for gpu in cfg.gpus])

    # 如果启用了调试模式，设置 Python 的断点调试器
    if cfg.debug:
        os.environ["PYTHONBREAKPOINT"] = "pdbr.set_trace"

    # if cfg.trained_model_dir.endswith(cfg.exp_name):
    #     pass
    # else:
    #     if len(cfg.exp_name_tag) != 0:
    #         cfg.exp_name +=  ('_' + cfg.exp_name_tag)
    #     cfg.exp_name = cfg.exp_name.replace('gitbranch', os.popen('git describe --all').readline().strip()[6:])
    #     cfg.exp_name = cfg.exp_name.replace('gitcommit', os.popen('git describe --tags --always').readline().strip())
    #     print('EXP NAME: ', cfg.exp_name)
    #     cfg.trained_model_dir = os.path.join(cfg.trained_model_dir, cfg.task, cfg.exp_name)
    #     cfg.record_dir = os.path.join(cfg.record_dir, cfg.task, cfg.exp_name)
    #     cfg.result_dir = os.path.join(cfg.result_dir, cfg.task, cfg.exp_name)
    #     cfg.local_rank = args.local_rank
    #     modules = [key for key in cfg if '_module' in key]
    #     for module in modules:
    #         cfg[module.replace('_module', '_path')] = cfg[module].replace('.', '/') + '.py'

    # 获取当前工作目录
    cur_workspace = os.environ['PWD']

    # 处理模型的路径
    if cfg.model_path == '':
        # 如果 model_path 为空，则根据任务名称和实验名称生成默认路径
        cfg.model_path = os.path.join('output', cfg.task, cfg.exp_name)
    
    # 如果 model_path 是相对路径，则将其转换为绝对路径
    if not os.path.isabs(cfg.model_path):
        cfg.model_path = os.path.join(cfg.workspace, cfg.model_path)
        # 规范化路径格式（去除多余的路径分隔符等）
        cfg.model_path = os.path.normpath(cfg.model_path)
    
    # 如果 model_path 不存在，则尝试基于当前工作目录重新生成路径
    if not os.path.exists(cfg.model_path):
        # 获取 model_path 相对于 workspace 的相对路径
        relative_path = os.path.relpath(cfg.model_path, cfg.workspace)
        # 将相对路径与当前工作目录结合，生成新的绝对路径
        cfg.model_path = os.path.join(cur_workspace, relative_path)

    # 如果 model_path 已存在且当前模式为训练模式，则提示用户该路径会被覆盖
    if os.path.exists(cfg.model_path) and cfg.mode == 'train':
        print('Model path already exists, this would override original model. 模型路径已经存在，这将覆盖原始模型。')
        print(f"model_path（模型路径）: {cfg.model_path}")

     # 在 model_path 下创建子目录
    cfg.trained_model_dir = os.path.join(cfg.model_path, 'trained_model')  # 训练模型的保存目录
    cfg.point_cloud_dir = os.path.join(cfg.model_path, 'point_cloud')      # 点云数据的保存目录

    # 处理数据路径
    if not os.path.isabs(cfg.source_path):
        # 如果 source_path 是相对路径，则将其转换为绝对路径
        cfg.source_path = os.path.join(cfg.workspace, cfg.source_path)
        # 规范化路径格式
        cfg.source_path = os.path.normpath(cfg.source_path)
    
    # 如果 source_path 不存在，则尝试基于当前工作目录重新生成路径
    if not os.path.exists(cfg.source_path):
        # 获取 source_path 相对于 workspace 的相对路径
        relative_path = os.path.relpath(cfg.source_path, cfg.workspace)
        # 将相对路径与当前工作目录结合，生成新的绝对路径
        cfg.source_path = os.path.join(cur_workspace, relative_path)
        # 如果路径仍然不存在，则进入调试模式
        if not os.path.exists(cfg.source_path):
            __import__('ipdb').set_trace()
            
    # 处理日志log的路径
    if cfg.record_dir is None:
        # 如果 record_dir 为空，则根据任务名称和实验名称生成默认路径
        # cfg.record_dir = os.path.join('output', 'record', cfg.task, cfg.exp_name)
        # 生成时间戳，格式如 20240101-120000
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        # 将时间戳添加到目录名中
        cfg.record_dir = os.path.join('output', 'record', cfg.task, cfg.exp_name, timestamp)
    
    # 如果 record_dir 是相对路径，则将其转换为绝对路径
    if not os.path.isabs(cfg.record_dir):
        cfg.record_dir = os.path.join(cfg.workspace, cfg.record_dir)
        # 规范化路径格式
        cfg.record_dir = os.path.normpath(cfg.record_dir)
    
    # 如果 record_dir 不存在，则尝试基于当前工作目录重新生成路径
    if not os.path.exists(cfg.record_dir):
        # 获取 record_dir 相对于 workspace 的相对路径
        relative_path = os.path.relpath(cfg.record_dir, cfg.workspace)
        # 将相对路径与当前工作目录结合，生成新的绝对路径
        cfg.record_dir = os.path.join(cur_workspace, relative_path)
    
def make_cfg(cfg, args):
    # 定义一个内部函数，用于递归加载配置文件
    def merge_cfg(cfg_file, cfg):
        # 打开配置文件并加载内容
        with open(cfg_file, 'r') as f:
            current_cfg = yacs.load_cfg(f)
        # 如果配置文件中包含 parent_cfg 字段，则递归加载父配置文件
        if 'parent_cfg' in current_cfg.keys():
            cfg = merge_cfg(current_cfg.parent_cfg, cfg)
            # 将当前配置文件的参数合并到 cfg 中
            cfg.merge_from_other_cfg(current_cfg)
        else:
            # 如果没有 parent_cfg，则直接合并当前配置文件的参数
            cfg.merge_from_other_cfg(current_cfg)
        # 打印当前加载的配置文件路径
        print(cfg_file)
        return cfg

    # 递归加载并合并配置文件
    cfg_ = merge_cfg(args.config, cfg)

    # 合并命令行参数
    try:
        # 如果命令行参数中包含 other_opts，则只合并 other_opts 之前的参数
        index = args.opts.index('other_opts')
        cfg_.merge_from_list(args.opts[:index])
    except:
        # 如果没有 other_opts，则合并所有命令行参数
        cfg_.merge_from_list(args.opts)

    # 调用 parse_cfg 函数，进一步解析和初始化配置参数
    parse_cfg(cfg_, args)

    # 返回最终的配置对象
    return cfg_

def save_cfg(cfg, model_dir, epoch=0):
    # 导入 redirect_stdout，用于将输出重定向到文件
    from contextlib import redirect_stdout

    # 创建模型目录（如果不存在）
    os.system('mkdir -p {}'.format(model_dir))

    # 在模型目录下创建 configs 子目录（如果不存在）
    cfg_dir = os.path.join(model_dir, 'configs')
    os.system('mkdir -p {}'.format(cfg_dir))

    # 生成配置文件的路径，文件名为 config_<epoch>.yaml
    cfg_path = os.path.join(cfg_dir, f'config_{epoch:06d}.yaml')

    # 将配置内容写入文件
    with open(cfg_path, 'w') as f:
        # 使用 redirect_stdout 将配置内容重定向到文件
        with redirect_stdout(f): print(cfg.dump())
        
    # 打印保存配置文件的路径
    print(f'保存输入的config到路径: {cfg_path}')
