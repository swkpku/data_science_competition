import os
import operator
import schedule
import time

num_checkpoint_to_save = 3

def delete_checkpoint():
    files = [f for f in os.listdir('.') if os.path.isfile(f)]

    checkpoint_files = {}

    for f in files:
        if f.endswith('.pth.tar'):
            checkpoint_files[f] =  os.path.getmtime(f)


    sorted_checkpoint_file = sorted(checkpoint_files.items(), key=operator.itemgetter(1))

    for i, (f, _) in enumerate(sorted_checkpoint_file):
        if i >= len(sorted_checkpoint_file) - num_checkpoint_to_save:
            break
        os.remove(f)

schedule.every(1).minutes.do(delete_checkpoint)

while 1:
    schedule.run_pending()
    time.sleep(1)
