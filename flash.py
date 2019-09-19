from concurrent.futures import ThreadPoolExecutor, as_completed


def flash(fn, args_list, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(fn, args): args for args in args_list}
        futures = as_completed(future_to_url)
        return futures
