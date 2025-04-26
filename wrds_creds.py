import os

def read_pgpass():
    pgpass_path = os.path.expanduser("~/.pgpass")
    #print(f"Checking .pgpass at: {pgpass_path}")
    with open(pgpass_path, "r") as f:
        #print(f.read())
        uname=f.read().split(':')[-2]
        pword=f.read().split(':')[-1]
    return uname,pword