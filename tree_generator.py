# %%
import os
from os import listdir
from os.path import isdir

basic_link = "\u251c\u2500"  # ├─
last_link = "\u2514\u2500"  # └─
prefix_list = ("\u2502   ", "    ")  # │


class TreeGenerator:
    def __init__(self):
        self.tree = []
        return

    def _get_prefix(self, last_folder_list):
        prefix = ""
        for last_folder in last_folder_list:
            prefix += prefix_list[last_folder]

        return prefix

    def _get_contents_info(self, folder):
        contents = [[], []]  # files, folders

        for content in listdir(f"./{folder}"):
            is_folder = isdir(f"{folder}/{content}")
            contents[is_folder].append(content)

        files, folders = contents
        contents = sorted(folders) + sorted(files)

        return contents, len(contents), len(folders), len(files)

    def _get_tree_list(self, folder, depth, last_folder_list: list = []):
        contents = listdir(f"./{folder}")
        prefix = self._get_prefix(last_folder_list)
        contents, contents_count, folders_count, files_count = self._get_contents_info(folder)

        for index, content in enumerate(contents):
            last_content = index == contents_count - 1
            link = last_link if last_content is True else basic_link

            if isdir(f"{folder}/{content}") is True:
                self.tree.append(f"{prefix}{link}{content}")

                last_content = [index == folders_count - 1 and files_count == 0]
                self._get_tree_list(f"{folder}/{content}", depth + 1, last_folder_list + last_content)

            else:
                self.tree.append(f"{prefix}{link}{content}")

        return self.tree

    def get_tree(self, folder, save=False):
        self.tree.clear()
        top_dir = os.path.basename(os.path.abspath(folder))

        self.tree.append(top_dir)
        tree = self._get_tree_list(folder, 0)
        print("\n".join(tree))

        if save is True:
            with open(f"{folder}/tree.txt", "w") as f:
                for line in tree:
                    f.write(line + "\n")

        return self.tree


tree_generator = TreeGenerator()
tree = tree_generator.get_tree(".", save=True)
