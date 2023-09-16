import sys
import importlib

def main():
    if len(sys.argv) < 2:
        print("Usage: wrapper.py module.path [args...]")
        sys.exit(1)


    backup_argv = sys.argv
    sys.argv = list(sys.argv)

    # if '--' in sys.argv[1]:
        # for i in range(1, len(sys.argv)):
            # if '--' not in sys.argv[i]:
                # module = sys.argv.pop(i)
                # sys.argv.insert(1, module)
                # break
    # for i in range(len(range(len(sys.argv)))):
        # if 'local-rank' in sys.argv[i]:
            # sys.argv[i] = sys.argv[i].replace('local-rank', 'local_rank')
            # option_name, rank = sys.argv[i].split("=")
            # sys.argv[i] = option_name
            # sys.argv.insert(i+1, rank)
            # break

    
    module_path = sys.argv[1]
    module_parts = module_path.split(".")

    
    # 使用 sys.argv 的方式模仿命令行参数
    sys.argv = [module_path] + sys.argv[2:]  # 从 args 中删除模块名称，将其他参数前移

    try:
        if len(module_parts) > 1:
            pkg = ".".join(module_parts[:-1])
            mod = module_parts[-1]
            print(pkg, mod)
            module = importlib.import_module(f".{mod}", package=pkg)
        else:
            module = importlib.import_module(module_path)
        
        # 模拟模块作为脚本运行
        module.__name__ = "__main__"
        exec(open(module.__file__).read(), module.__dict__)
        
    except ImportError:
        print(f"Error: Unable to import module '{module_path}'")
        raise
    finally:
        sys.argv = backup_argv

if __name__ == "__main__":
    main()
