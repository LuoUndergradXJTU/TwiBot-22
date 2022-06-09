## Files 
Each dataset contains `node.json`, `label.csv`, `split.csv` and `edge.csv` (for datasets with graph structure).

### `node.json` or `{tweet, user, list, hashtag}.json`
This file contains twitter user information (for non-graph datasets) or entities (including users, tweets, lists and etc. See [here](statistics.md) for details).

The node information of TwiBot-22 is too big for a single file, so we use `tweet_0.json` to `tweet_8.json`, `user.json`, `list.json`, and `hashtag.json` to store them.

### `split.csv`
This file contains data split information, where the first column (id) is the user id and the second column (split) is the corresponding split (train, valid or test).

### `label.csv`
This file contains the ground truth labels, where the first column (id) is the user id and the second column (label) is the corresponding label (human or bot).

### `edge.csv`
This file contains relations of entities appear in `node.json`. Each of the entries contains source_id, target_id and relation type. See [here](statistics.md) for a detailed description for each relation type.

## For New Datsets
We welcome new bot detection datasets. Please convert the original dataset to the files and schema defined above. PR's welcome.
