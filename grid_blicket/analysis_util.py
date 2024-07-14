from __future__ import annotations
import numpy as np
import cv2
import torch
import os 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import imageio
from minigrid.core.constants import TILE_PIXELS, COLORS, COLOR_NAMES, COLOR_TO_IDX, IDX_TO_COLOR, OBJECT_TO_IDX, IDX_TO_OBJECT, STATE_TO_IDX

def plot_cos_scores(cos_scores, dir_path):
    # cos_scores: len_epi, n_rollouts, n_prototypes
    # for each prototype, plot the cosine similarity scores over len_epi. one curve per rollout
    n_prototypes = cos_scores.shape[2]
    n_rollouts = cos_scores.shape[1]
    len_epi = cos_scores.shape[0]
    for i in range(n_prototypes):
        plt.figure()
        for j in range(n_rollouts):
            plt.plot(cos_scores[:, j, i], label=f'rollout_{j}')
        plt.title(f'Prototype {i}')
        plt.xlabel('Time in episode')
        plt.ylabel('cosine similarity')
        plt.legend()
        plt.savefig(os.path.join(dir_path, f'prototype_{i}_cos_scores.png'))

def render_tile(ax, pos, object_idx, color_idx, state):
    """
    Renders a single tile at the given position in the given Axes object.
    """
    x, y = pos
    color = COLORS[IDX_TO_COLOR[color_idx]] / 255  # Normalize color to [0, 1] range for matplotlib
    object_type = IDX_TO_OBJECT[object_idx]

    if object_type == 'wall':
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=color, facecolor=color)
        ax.add_patch(rect)
    elif object_type == 'floor':
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='black', facecolor=color)
        ax.add_patch(rect)
    elif object_type == 'door':
        if state == STATE_TO_IDX['open']:
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        elif state == STATE_TO_IDX['closed']:
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            rect = patches.Rectangle((x, y), 1, 0.5, linewidth=1, edgecolor=color, facecolor=color)
            ax.add_patch(rect)
        elif state == STATE_TO_IDX['locked']:
            rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor=color, facecolor=color)
            ax.add_patch(rect)
    elif object_type == 'key':
        circle = patches.Circle((x + 0.5, y + 0.5), 0.3, linewidth=1, edgecolor=color, facecolor=color)
        ax.add_patch(circle)
    elif object_type == 'ball':
        circle = patches.Circle((x + 0.5, y + 0.5), 0.3, linewidth=1, edgecolor=color, facecolor=color)
        ax.add_patch(circle)
    elif object_type == 'box':
        rect = patches.Rectangle((x + 0.2, y + 0.2), 0.6, 0.6, linewidth=1, edgecolor=color, facecolor=color)
        ax.add_patch(rect)
    elif object_type == 'goal':
        star = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=5, radius=0.3, orientation=np.pi/4, edgecolor=color, facecolor=color)
        ax.add_patch(star)
    elif object_type == 'lava':
        rect = patches.Rectangle((x, y), 1, 1, linewidth=1, edgecolor='red', facecolor='orange')
        ax.add_patch(rect)
    elif object_type == 'agent':
        triangle = patches.RegularPolygon((x + 0.5, y + 0.5), numVertices=3, radius=0.4, orientation=np.pi/2, edgecolor=color, facecolor=color)
        ax.add_patch(triangle)

def render_obs(obs):
    """
    Renders the (3, 7, 7) observation, where each tile of the 7x7 grid is encoded as a 3 dimensional tuple: (OBJECT_IDX, COLOR_IDX, STATE), into RGB image.
    
    :param obs: Observation tensor of shape (3, 7, 7)
    :return: 3-channel RGB image of appropriate shape to contain information about the object, its color, and its state.
    """
    height, width = obs.shape[1], obs.shape[2]
    fig, ax = plt.subplots(1, figsize=(width, height))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')

    for i in range(height):
        for j in range(width):
            object_idx, color_idx, state = obs[:, i, j]
            render_tile(ax, (j, height - 1 - i), object_idx, color_idx, state)

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return image


def plot_and_save_obs_image(obs_image, save_path=None):
    """
    Plots the observation image and optionally saves it to a file.
    
    Args:
    obs_image: The RGB image array returned by render_obs
    save_path: Optional; if provided, the path where the image will be saved
    """
    # Convert from BGR to RGB if using OpenCV
    obs_image_rgb = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(obs_image_rgb)
    plt.axis('off')  # Turn off axis numbers
    plt.title('Observation Image')
    
    # Save the image if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        print(f"Image saved to {save_path}")

def save_episode_as_gif(episode_obs, filename='episode.gif', duration=100):
    """
    Saves an episode of observations as a GIF.
    
    Args:
    episode_obs: numpy array of shape (T, 3, 7, 7) containing T observations
    filename: string, the name of the output GIF file
    fps: int, frames per second for the GIF
    """
    # Create a list to store rendered frames
    frames = []
    
    # Render each observation and append to frames
    for obs in tqdm(episode_obs, desc="Rendering frames"):
        obs_image = render_obs(obs)
        frames.append(obs_image)
    
    # Save the frames as a GIF
    imageio.mimsave(filename, frames, duration=duration)
    print(f"GIF saved as {filename}")


def plot_high_cos_obs(cos_scores, obs_batch, threshold, save_path=None):
    """
    Generate one figure for each prototype. 
    Each rollout gets a row in the plot.
    Plots the observations with high cosine similarity scores.
    
    Args:
    cos_scores: numpy array of shape (len_episode, num_rollouts, num_prototype) containing cosine similarity scores
    obs_batch: numpy array of shape (len_episode, num_rollouts, num_prototype, 3, 7, 7) containing abstract observations of shape (3, 7, 7)
    threshold: float, the threshold above which to consider a cosine similarity score as high
    save_path: Optional; if provided, the path where the images will be saved
    """
    len_episode, num_rollouts, num_prototype = cos_scores.shape
    
    for p in range(num_prototype):
        # Count the maximum number of high cosine similarity scores for any rollout
        max_high_cos = max(np.sum(cos_scores[:, r, p] > threshold) for r in range(num_rollouts))
        
        # Create a figure with a row for each rollout
        fig = plt.figure(figsize=(4 * max_high_cos, 4 * num_rollouts))
        gs = GridSpec(num_rollouts, max_high_cos, figure=fig)
        
        for r in range(num_rollouts):
            high_cos_indices = np.where(cos_scores[:, r, p] > threshold)[0]
            
            for i, idx in enumerate(high_cos_indices):
                ax = fig.add_subplot(gs[r, i])
                obs_image = render_obs(obs_batch[idx, r, p])
                obs_image_rgb = cv2.cvtColor(obs_image, cv2.COLOR_BGR2RGB)
                ax.imshow(obs_image_rgb)
                ax.axis('off')
                ax.set_title(f'Step {idx}\nCos: {cos_scores[idx, r, p]:.2f}')
            
            # If there are fewer high cosine scores than max_high_cos, fill with empty subplots
            for i in range(len(high_cos_indices), max_high_cos):
                ax = fig.add_subplot(gs[r, i])
                ax.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'Prototype {p+1}', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(os.path.join(save_path, f'prototype_{p+1}.png'), bbox_inches='tight', dpi=300)
            print(f"Image for Prototype {p+1} saved to {os.path.join(save_path, f'prototype_{p+1}.png')}")
        
        # plt.show()
        # plt.close()