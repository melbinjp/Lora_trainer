import ctypes
import psutil
import os

def flush_ram():
    print("--- Aggressive RAM Cleaner ---")
    initial_mem = psutil.virtual_memory().available / (1024**3)
    print(f"Initial Available: {initial_mem:.2f} GB")
    
    # Enable SeDebugPrivilege if running as admin (optional but helps)
    # For now, we just try to flush everything we can.
    
    count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # PROCESS_SET_QUOTA (0x0100) | PROCESS_VM_OPERATION (0x0008)
            handle = ctypes.windll.kernel32.OpenProcess(0x0108, False, proc.info['pid'])
            if handle:
                # -1, -1 tells Windows to empty the working set
                ctypes.windll.kernel32.SetProcessWorkingSetSize(handle, -1, -1)
                ctypes.windll.kernel32.CloseHandle(handle)
                count += 1
        except:
            pass
            
    final_mem = psutil.virtual_memory().available / (1024**3)
    print(f"Flushed {count} processes.")
    print(f"Final Available: {final_mem:.2f} GB")
    print(f"Gained: {final_mem - initial_mem:.2f} GB")

if __name__ == "__main__":
    flush_ram()
