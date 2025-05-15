# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 19:31:08 2024

@author: sjgauva
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree
from sklearn.decomposition import PCA
#%% Create visualizations to show alignment process

def generate_example_point_cloud(n_points=100, angle_deg=0, noise_level=0.1):
    """Generate a synthetic cylindrical point cloud at a given angle"""
    # Create points along a cylinder wall
    theta = np.linspace(0, 2*np.pi, n_points)
    z = np.linspace(-1, 1, n_points)
    r = 1.0 + noise_level * np.random.randn(n_points)
    
    # Generate cylindrical coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Stack coordinates
    points = np.column_stack((x, y, z))
    
    # Apply rotation if angle isn't 0
    if angle_deg != 0:
        angle_rad = np.radians(angle_deg)
        rotation_matrix = create_rotation_matrix(angle_rad)
        points = apply_rotation(points, rotation_matrix)
    
    return points

def create_rotation_matrix(angle_rad):
    """Create rotation matrix around Y-axis"""
    return np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])

def apply_rotation(points, rotation_matrix):
    """Apply rotation matrix to points"""
    return np.dot(points, rotation_matrix.T)

def plot_point_clouds(clouds, titles=None, fig=None):
    """Plot multiple point clouds in 3D"""
    if fig is None:
        fig = plt.figure(figsize=(15, 5))
    
    n_clouds = len(clouds)
    for i, points in enumerate(clouds):
        ax = fig.add_subplot(1, n_clouds, i+1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', alpha=0.6)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if titles and i < len(titles):
            ax.set_title(titles[i])
    plt.tight_layout()
    plt.show()

def demonstrate_alignment_process():
    """Demonstrate each step of the point cloud alignment process"""
    # Step 1: Generate example point clouds at different angles
    print("Step 1: Generating example point clouds...")
    cloud_neg60 = generate_example_point_cloud(angle_deg=-60)
    cloud_0 = generate_example_point_cloud(angle_deg=0)
    cloud_pos60 = generate_example_point_cloud(angle_deg=60)
    
    plot_point_clouds(
        [cloud_neg60, cloud_0, cloud_pos60],
        ["Scan at -60°", "Reference Scan (0°)", "Scan at +60°"]
    )
    
    # Step 2: Initial rotation to vertical reference frame
    print("\nStep 2: Rotating scans to vertical reference frame...")
    rotation_matrix_neg60 = create_rotation_matrix(np.radians(60))  # Note: opposite angle
    rotation_matrix_pos60 = create_rotation_matrix(np.radians(-60))
    
    cloud_neg60_rotated = apply_rotation(cloud_neg60, rotation_matrix_neg60)
    cloud_pos60_rotated = apply_rotation(cloud_pos60, rotation_matrix_pos60)
    
    print("Rotation Matrix for -60° scan:")
    print(rotation_matrix_neg60)
    
    plot_point_clouds(
        [cloud_neg60_rotated, cloud_0, cloud_pos60_rotated],
        ["Rotated -60° Scan", "Reference Scan", "Rotated +60° Scan"]
    )
    
    # Step 3: Calculate centroids (in this case, we'll use the points themselves)
    print("\nStep 3: Using points as centroids...")
    # In a real mesh, you'd calculate face centroids
    centroids_neg60 = cloud_neg60_rotated
    centroids_0 = cloud_0
    centroids_pos60 = cloud_pos60_rotated
    
    # Step 4: Build KDTree and find nearest neighbors
    print("\nStep 4: Building KDTree and finding nearest neighbors...")
    tree = KDTree(centroids_0)  # Reference cloud
    
    # Find nearest neighbors for -60° scan
    distances_neg60, indices_neg60 = tree.query(centroids_neg60)
    matched_points_ref_neg60 = centroids_0[indices_neg60]
    
    print("Average distance to nearest neighbors (-60° scan):", np.mean(distances_neg60))
    
    # Step 5: PCA Alignment
    print("\nStep 5: PCA Alignment...")
    
    def align_with_pca(source_points, target_points):
        # Compute PCA for both point sets
        pca_source = PCA(n_components=3)
        pca_target = PCA(n_components=3)
        
        # Fit PCAs
        pca_source.fit(source_points)
        pca_target.fit(target_points)
        
        # Get principal components
        components_source = pca_source.components_
        components_target = pca_target.components_
        
        # Calculate rotation matrix between principal components
        rotation = np.dot(components_target.T, components_source)
        
        # Apply rotation
        aligned_points = np.dot(source_points, rotation)
        
        return aligned_points, rotation
    
    # Align -60° scan
    aligned_neg60, rotation_neg60 = align_with_pca(centroids_neg60, centroids_0)
    print("PCA Rotation Matrix for -60° scan:")
    print(rotation_neg60)
    
    # Align +60° scan
    aligned_pos60, rotation_pos60 = align_with_pca(centroids_pos60, centroids_0)
    
    # Step 6: Show final aligned results
    print("\nStep 6: Showing final aligned results...")
    plot_point_clouds(
        [aligned_neg60, cloud_0, aligned_pos60],
        ["Aligned -60° Scan", "Reference Scan", "Aligned +60° Scan"]
    )
    
    # Calculate and display alignment metrics
    print("\nAlignment Metrics:")
    final_distances_neg60, _ = KDTree(cloud_0).query(aligned_neg60)
    final_distances_pos60, _ = KDTree(cloud_0).query(aligned_pos60)
    
    print(f"Mean alignment error for -60° scan: {np.mean(final_distances_neg60):.4f}")
    print(f"Mean alignment error for +60° scan: {np.mean(final_distances_pos60):.4f}")
    
    # Step 7: Merge point clouds
    print("\nStep 7: Merging point clouds...")
    merged_cloud = np.vstack([aligned_neg60, cloud_0, aligned_pos60])
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, projection='3d')
    ax.scatter(merged_cloud[:, 0], merged_cloud[:, 1], merged_cloud[:, 2], 
              c='b', alpha=0.6)
    ax.set_title("Final Merged Point Cloud")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__ == "__main__":
    demonstrate_alignment_process()