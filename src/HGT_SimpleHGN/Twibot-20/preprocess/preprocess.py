import torch
from dataset import hetero_graph_vectorize

graph, uid_to_user_index, tid_to_tweet_index, train_uid_with_label, valid_uid_with_label, test_uid_with_label = hetero_graph_vectorize(include_node_feature=True, dataset="Twibot-20", server_id="209")

torch.save(graph, "./graph.pt")
torch.save(uid_to_user_index, "./uid_to_user_index.pt")
torch.save(tid_to_tweet_index, "./tid_to_tweet_index.pt")
torch.save(train_uid_with_label, "./train_uid_with_label.pt")
torch.save(valid_uid_with_label, "./valid_uid_with_label.pt")
torch.save(test_uid_with_label,"./test_uid_with_label.pt")
