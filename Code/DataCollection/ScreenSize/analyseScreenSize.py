import json

with open("screenSizeResults.json") as data_file:
    results = json.load(data_file)

# Set some starting values
xMax = yMax = 0
xMin = yMin = 10000
all4Channels = True

for g in results:
    # Check the number of channels printing out if it is different
    if not g["channels"] == 4:
        print(g["Game"] + " has " + str(g["channels"]) + " channels")
        all4Channels = False

    # Check the x size
    if g["xPixels"] > xMax:
        xMax = g["xPixels"]
    if g["xPixels"] < xMin:
        xMin = g["xPixels"]

    # Check the y size
    if g["yPixels"] > yMax:
        yMax = g["yPixels"]
    if g["yPixels"] < yMin:
        yMin = g["yPixels"]

# Report final Coord Size
print("  | Min | Max")
print("--|-----|-----")
print("X | " + str(xMin) + "  | " + str(xMax))
print("Y | " + str(yMin) + "  | " + str(yMax))
# Report if all the games have 4 channels
if all4Channels:
    print("All games have 4 channels")
