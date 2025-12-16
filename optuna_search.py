"""Vision Token Pruning - Optuna超参数搜索脚本

使用Manager的Optuna模式进行超参数搜索。
通过字典覆盖方式启用Optuna搜索，无需单独的配置文件。
"""

from engine.configs.loader import load_config
from engine.managers.loader import load_manager

# 复用main.py中的函数
from main import preload_fn, run_fn


def main():
    """主函数 - Optuna超参数搜索"""
    # 使用字典覆盖方式启用Optuna搜索（只修改最小必要项）
    override_dict = {
        "manager_settings": {
            "mode": "optuna",
        },
        "search_settings": {
            "enable": True,
        },
        "global_settings": {
            "study_name": "vision_token_pruning"
        }
    }

    # 加载配置，使用字典覆盖
    config = load_config(
        override_file="configs/vision_token_pruning.yaml",
        override_dict=override_dict
    )
    
    logger = config["logger"]
    
    logger.info("=" * 60)
    logger.info("Vision Token Pruning - Optuna超参数搜索")
    logger.info("=" * 60)
    logger.info(f"Study名称: {config['search_settings']['study_name']}")
    logger.info(f"试验次数: {config['search_settings']['n_trials']}")
    logger.info(f"并行worker: {config['manager_settings']['num_subtasks']}")
    logger.info("=" * 60)
    
    # 创建Manager（Optuna模式）
    # 注意：不需要提供task_generator_fn和result_handler_fn
    # Manager会自动使用内置的Optuna版本
    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn
    )
    
    # 启动搜索
    manager.start()
    manager.wait()
    
    # 获取最优结果
    if hasattr(manager, 'best_trial_info'):
        best = manager.best_trial_info
        logger.info("=" * 60)
        logger.info("超参数搜索完成!")
        logger.info(f"最优Trial: {best['trial_number']}")
        logger.info(f"最优Score: {best['best_score']:.4f}")
        logger.info(f"最优参数:")
        for k, v in best['best_params'].items():
            logger.info(f"  {k}: {v}")
        logger.info(f"最优参数已保存到: {config['global_settings']['task_dir']}/best_params.json")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("搜索完成，但未找到最优试验信息")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
