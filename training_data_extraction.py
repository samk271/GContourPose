import re

f = open("bowl_5_training_log.txt")

pattern = r"Epoch (\d+) \|\| Average Loss: ([\d\.eE\+\-]+) \|\|.*\|\| Average Overlap: ([\d\.eE\+\-]+)"

out = open("bowl_5.csv", "w")
for line in f:
    match = re.search(pattern, line)
    if match:
        out.write("{},{},{}\n".format(match.group(1),match.group(2),match.group(3)))