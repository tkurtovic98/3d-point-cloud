import os

def start():
    path = "/home/tomislav/Downloads/Garfield"

    counter = 0

    for file in os.listdir(path):
        if not file.endswith(".ply"):
            continue

        file = path + "/" + file

        os.rename(file, f'{file.split("_")[0]}_{counter}.ply')
        counter +=1

if __name__ == "__main__":
    start()
