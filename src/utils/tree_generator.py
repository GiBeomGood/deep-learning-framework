from pathlib import Path

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

    def _get_contents_info(self, folder: Path):
        contents = [[], []]  # files, folders

        # Iterate over directory contents using pathlib
        for content in folder.iterdir():
            # Check if it's a directory
            is_folder = content.is_dir()
            contents[is_folder].append(content.name)

        files, folders = contents
        # Sort folders and files
        contents = sorted(folders) + sorted(files)

        return contents, len(contents), len(folders), len(files)

    def _get_tree_list(self, folder: Path, depth, last_folder_list: list = []):
        # Determine prefix based on the last folder list
        prefix = self._get_prefix(last_folder_list)
        contents, contents_count, folders_count, files_count = self._get_contents_info(folder)

        for index, content_name in enumerate(contents):
            last_content = index == contents_count - 1
            link = last_link if last_content is True else basic_link

            content_path = folder / content_name
            if content_path.is_dir():
                self.tree.append(f"{prefix}{link}{content_name}")

                last_content_flag = [index == folders_count - 1 and files_count == 0]
                self._get_tree_list(content_path, depth + 1, last_folder_list + last_content_flag)

            else:
                self.tree.append(f"{prefix}{link}{content_name}")

        return self.tree

    def get_tree(self, folder: str | Path, save=False):
        self.tree.clear()
        folder_path = Path(folder)
        top_dir = folder_path.resolve().name

        self.tree.append(top_dir)
        tree = self._get_tree_list(folder_path, 0)
        print("\n".join(tree))

        if save is True:
            with (folder_path / "tree.txt").open("w") as f:
                for line in tree:
                    f.write(line + "\n")

        return self.tree


tree_generator = TreeGenerator()
tree = tree_generator.get_tree(".", save=True)