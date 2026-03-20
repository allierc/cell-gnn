import subprocess


def run_claude_cli(prompt, root_dir, max_turns=500, allowed_tools=None):
    """Run Claude CLI with real-time output streaming. Returns output text."""
    if allowed_tools is None:
        allowed_tools = ['Read', 'Edit', 'Write']
    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', str(max_turns),
        '--allowedTools',
        *allowed_tools,
    ]

    output_lines = []
    process = subprocess.Popen(
        claude_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    return ''.join(output_lines)
