import psutil

def list_top_mem():
    procs = []
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            procs.append((p.info['name'], p.info['memory_info'].rss))
        except:
            pass
            
    procs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Process Name':<30} | {'Memory (MB)':>12}")
    print("-" * 45)
    for name, rss in procs[:25]:
        print(f"{name:<30} | {rss/1024/1024:>12.2f}")

if __name__ == "__main__":
    list_top_mem()
