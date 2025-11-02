import yaml

class ParseConfig:
    """解析YAML配置文件的类
    
    属性:
        config (dict): 存储配置文件内容的字典
    
    方法:
        load_config(config_file): 加载YAML配置文件并返回配置字典
    """
    def __init__(self, config_file):
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        """加载YAML配置文件"""
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    
def main():
    config_parser = ParseConfig('config.yaml')
    config = config_parser.config
    print(config)
    
if __name__ == "__main__":
    main()