from pathlib import Path

app_path = Path('app.py')
code = app_path.read_text(encoding='utf-8')

# Find the start and end of HTML_DASHBOARD
start_idx = code.find('HTML_DASHBOARD = """<!DOCTYPE html>')
end_idx = code.find('"""', start_idx + 10) + 3
if start_idx != -1 and end_idx != -1:
    code = code[:start_idx] + code[end_idx:]

# Replace the handler usage
old_serve = 'self.wfile.write(HTML_DASHBOARD.encode("utf-8"))'
new_serve = '''with open(PROJECT_ROOT / "templates" / "index.html", "rb") as f:\n                    self.wfile.write(f.read())'''
code = code.replace(old_serve, new_serve)

app_path.write_text(code, encoding='utf-8')
print('Successfully refactored app.py')
