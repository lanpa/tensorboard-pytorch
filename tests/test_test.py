def test_test():
    from examples import demo
    from examples import demo_graph
    from examples import demo_embedding
    from examples import demo_custom_scalars
    from examples import demo_multiple_embedding
    from examples import demo_purge
    from examples import demo_matplotlib
    import subprocess
    subprocess.check_output(['flake8', 'tensorboardX'])
