from obj_init import create_universe, show_universe, show_clusters
from clustering import cluster_loop

K = create_universe()

show_clusters(cluster_loop(K, 3))
