import torch
import torch.nn.functional as F

def get_x_percent_boundary(distances, x):
  """ Returns the xth percentile of the distances
  
  Args:
      distances: list of distances
      x: int (percentile)
      
  Returns:
      float: xth percentile of the distances
  """
  distances.sort()
  length = len(distances)
  return distances[int(length * (x/100))]


def mahalanobis_distance(gaussian_mean, gaussian_covariance, flattened_feature_space):
  """ Returns the Mahalanobis distance between the feature space and the gaussian distribution
  
  Args:
      gaussian_mean: tensor of size 1024 x 1  (mean of the gaussian distribution)     
      gaussian_covariance: tensor of size 1024 x 1024  (covariance of the gaussian distribution)
      feature_space: tensor of size 1024 x 1  (flattened feature space)
    
  Returns:
      Mahalanobis distance: float
  """

  diff = torch.sub(flattened_feature_space, gaussian_mean)
  inv = torch.linalg.inv(gaussian_covariance)
  return (diff.view(1,-1) @ inv @ diff).item()

def is_sample_ood(flattened_feature_space, class_mean, class_covariance, class_boundary):
  """ Returns True if the sample is out of the class boundary
  
  Args:
      flattened_feature_space: tensor of size 1024 x 1  (flattened feature space)
      class_mean: tensor of size 1024 x 1  (mean of the gaussian distribution)     
      class_covariance: tensor of size 1024 x 1024  (covariance of the gaussian distribution)
      class_boundary: float (boundary of the class)

  Returns:  
      bool: True if the sample is out of the class boundary
  """
  return mahalanobis_distance(class_mean, class_covariance, flattened_feature_space) > class_boundary

def pool_and_flatten(feature_space, kernel_size, stride):
  """ Pool and flatten the feature space        
    Args:
        feature_space: tensor of size 1024 x 1  (flattened feature space)
        kernel_size: int (size of the kernel)
        stride: int (stride of the kernel)
        
    Returns:
        tensor: flattened feature space
  """
  feature_space = F.avg_pool2d(feature_space, kernel_size=kernel_size, stride=stride)

  # Continue pooling until the total size falls below 1e4
  while torch.numel(feature_space) >= 1e4:
      feature_space = F.avg_pool2d(feature_space, kernel_size=kernel_size, stride=stride)

  # Flatten the subspace
  feature_space = feature_space.view(-1)
  return feature_space