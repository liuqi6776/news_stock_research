import os
import re

def refactor_imports(directory):
    for root, dirs, files in os.walk(directory):
        if '.git' in dirs:
            dirs.remove('.git')
        if '__pycache__' in dirs:
            dirs.remove('__pycache__')
            
        for file in files:
            if file.endswith('.py'):
                path = os.path.join(root, file)
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Replace 'from infra_data.' with 'from infra_data.'
                new_content = re.sub(r'from data\.', 'from infra_data.', content)
                
                if new_content != content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"Refactored: {path}")

if __name__ == "__main__":
    refactor_imports(os.getcwd())
