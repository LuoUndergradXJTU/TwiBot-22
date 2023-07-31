import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.metrics import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

def evaluate(truth, pred, embedding):

    rans = rand_score(truth, pred)
    aran = adjusted_rand_score(truth, pred)

    mutu = mutual_info_score(truth, pred)
    nmut = normalized_mutual_info_score(truth, pred)
    amut = adjusted_mutual_info_score(truth, pred)

    homo = homogeneity_score(truth, pred)
    comp = completeness_score(truth, pred)
    vmea = v_measure_score(truth, pred, beta=1.0)

    fow = fowlkes_mallows_score(truth, pred)

    # Intrinsic measure
    sil = silhouette_score(embedding, pred, metric="euclidean")
    cal_har = calinski_harabasz_score(embedding, pred)
    dav = davies_bouldin_score(embedding, pred)

    # Store all above variables into a dictionary
    metrics = {'rand_score': rans,
                'adjusted_rand_score': aran,
                'mutual_info_score': mutu,
                'normalized_mutual_info_score': nmut,
                'adjusted_mutual_info_score': amut,
                'homogeneity_score': homo,
                'completeness_score': comp,
                'v_measure_score': vmea,
                'fowlkes_mallows_score': fow,
                'silhouette_score': sil,
                'calinski_harabasz_score': cal_har,
                'davies_bouldin_score': dav}
    
    return metrics
               
def cluster(output_df, num_clusters=2, config="3_layer_lr_00001_long_long_deep"):
    # Load tensors
    train_embedding = torch.load(f"output/{config}_train_embedding.pt",
                            map_location=torch.device('cpu')).detach().numpy() 
    train_label = torch.load(f"output/{config}_train_label.pt",
                        map_location=torch.device('cpu')).detach().numpy() 
    # test_pred = torch.load(f"output/{config}_{valid}_pred.pt",
    #                     map_location=torch.device('cpu')).detach().numpy() 

    val_embedding = torch.load(f"output/{config}_valid_embedding.pt",
                            map_location=torch.device('cpu')).detach().numpy() 
    val_label = torch.load(f"output/{config}_valid_label.pt",
                        map_location=torch.device('cpu')).detach().numpy() 
    
    test_embedding = torch.load(f"output/{config}_test_embedding.pt",
                            map_location=torch.device('cpu')).detach().numpy() 
    test_label = torch.load(f"output/{config}_test_label.pt",
                        map_location=torch.device('cpu')).detach().numpy() 
    
    # kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(train_embedding)
    train_pred = kmeans.predict(train_embedding)
    val_pred = kmeans.predict(val_embedding)
    test_pred = kmeans.predict(test_embedding)
    train_metrics = evaluate(train_label, train_pred, train_embedding)
    val_metrics = evaluate(val_label, val_pred, val_embedding)
    test_metrics = evaluate(test_label, test_pred, test_embedding)
    
    # Unroll all the metrics in the dictionary into a list
    train_metrics_list = [train_metrics[key] for key in train_metrics]
    val_metrics_list = [val_metrics[key] for key in val_metrics]
    test_metrics_list = [test_metrics[key] for key in test_metrics]

    # Combine all the metrics into a list
    combined_metrics_list = [num_clusters] + train_metrics_list + val_metrics_list + test_metrics_list

    # Add combined_metrics_list to output_df
    output_df.loc[len(output_df)] = combined_metrics_list
    return output_df

if __name__ == "__main__":
    combined_metrics_list = ['num_clusters','train_rand_score', 'train_adjusted_rand_score', 'train_mutual_info_score', 'train_normalized_mutual_info_score', 'train_adjusted_mutual_info_score', 'train_homogeneity_score', 'train_completeness_score', 'train_v_measure_score', 'train_fowlkes_mallows_score', 'train_silhouette_score', 'train_calinski_harabasz_score', 'train_davies_bouldin_score', 'val_rand_score', 'val_adjusted_rand_score', 'val_mutual_info_score', 'val_normalized_mutual_info_score', 'val_adjusted_mutual_info_score', 'val_homogeneity_score', 'val_completeness_score', 'val_v_measure_score', 'val_fowlkes_mallows_score', 'val_silhouette_score', 'val_calinski_harabasz_score', 'val_davies_bouldin_score', 'test_rand_score', 'test_adjusted_rand_score', 'test_mutual_info_score', 'test_normalized_mutual_info_score', 'test_adjusted_mutual_info_score', 'test_homogeneity_score', 'test_completeness_score', 'test_v_measure_score', 'test_fowlkes_mallows_score', 'test_silhouette_score', 'test_calinski_harabasz_score', 'test_davies_bouldin_score']
    output_df = pd.DataFrame(columns=combined_metrics_list)
    config = "3_layer_lr_00001_long_long_deep"
    for i in range(2, 100):
        output_df = cluster(output_df, num_clusters=i, config=config)
    output_df.to_csv(f"output/{config}_cluster.csv", index=False)
    print("Done!")