from segmentation.evaluation_utils import compare_cross_graphs, compare_simple_graphs

RESULTS_FILES = [
        "C:\\GitHub\\active_learning\\segmentation\\tests\\graphs\\exp7_area_less\\eval_random.txt",
        "C:\\GitHub\\active_learning\\segmentation\\tests\\graphs\\exp7_area_less\\eval_area.txt",
        "C:\\GitHub\\active_learning\\segmentation\\tests\\graphs\\exp7_area_less\\eval_area_inv.txt",
    ]
SAVE_PATH = "C:\\GitHub\\active_learning\\segmentation\\tests\\graphs\\exp4\\results.png"

if __name__ == "__main__":
    #compare_cross_graphs(RESULTS_FILES, ["Random", "Margin", "Entropy"], "acc")
    compare_simple_graphs(RESULTS_FILES, ["Random", "Area", "Area Inv"], "acc", SAVE_PATH)
