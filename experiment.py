#%% [markdown]
# This works from a Paperspace Jupyter notebook

import os

def install_cairo():
  
    def install_not_needed():
        try:
            import cairo
            return(True)
        except ImportError:
            pass
            return(False)

    def printlog(cmdout,log): 
        for line in cmdout:
            print(line, file = log)

    if install_not_needed():
        print("All requirements met, install not needed")
        return

    # Setup installation log
    if os.path.exists("setup.log"):
        os.remove("setup.log")  
    log = open("setup.log","a")
      
    print("\nSetup required ", file = log)

    from datetime import datetime

    print(datetime.utcnow(), file = log)
    import sys
    print("Log Python Version", file = log)
    print(sys.version, file=log)
    cmdout = !uname -a
    print("ID OS version", file = log)
    printlog(cmdout, log)
    print("Updating apt-get", file = log)  
    cmdout = !apt-get update    
    print("Installing cairo", file = log)  
    cmdout = !apt-get -y install libcairo2-dev
    printlog(cmdout, log)
    cmdout = !pip install pycairo
    printlog(cmdout, log)

    if install_not_needed():
        print("Install finished without errors")
    else:
        print("Install failed, check setup.log")

#%% [markdown]
# Load RPM

# %load RPM.py
