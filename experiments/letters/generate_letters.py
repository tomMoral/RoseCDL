# This script gives the ICML letters
# And generates a 2D dataset using the letters
# as the dictionary.


import numpy as np


def create_i(mean=0.5, std=0.01):
    value = np.random.normal(mean, std)

    # Create a 29x25 array of zeros
    image = np.zeros((29, 25))

    # Set the pixels of the letter "I" to value
    image[2:27, 11:15] = value

    image[0:4, 5:21] = value
    image[24:28, 5:21] = value

    return image


def create_c(height=29, width=25, mean=0.5, std=0.1):
    """
    Create a binary image of the letter C with moderate smoothing

    Parameters:
    height (int): Height of the image
    width (int): Width of the image
    mean (float): Mean value of the letter C
    std (float): Standard deviation of the letter C

    Returns:
    numpy.ndarray: Binary image of the letter C
    """
    value = np.random.normal(mean, std)

    # Create an empty array
    image = np.zeros((height, width), dtype=np.float64)

    # Center of the image
    center_y = height / 2
    center_x = width / 2

    # Radius and thickness parameters
    radius_x = width * 0.4
    radius_y = height * 0.4
    thickness = width * 0.06

    # Create C shape
    for y in range(height):
        for x in range(width):
            # Normalized coordinates
            norm_x = (x - center_x) / radius_x
            norm_y = (y - center_y) / radius_y

            # Distance from center
            dist = np.sqrt(norm_x**2 + norm_y**2)

            # Angle calculation
            angle = np.arctan2(norm_y, norm_x)

            # Check if point is within the C shape
            if (
                dist >= 1 - thickness / radius_x
                and dist <= 1 + thickness / radius_x
                and (angle > np.pi / 4 or angle < -np.pi / 4)
            ):
                image[y, x] = value

    return image


def create_m(mean=0.5, std=0.01):
    value = np.random.normal(mean, std)

    # Create a 29x25 array of zeros
    image = np.zeros((29, 25))

    # Set the pixels of the letter "M" to value
    image[0:28, 2:6] = value
    image[0:28, 18:22] = value

    for i in range(6, 18):
        x = min((i - 6), (18 - i))
        image[x : x + 5, i] = value

    return image


def create_l(mean=0.5, std=0.01):
    value = np.random.normal(mean, std)

    image = np.zeros((29, 25))

    image[2:27, 5:9] = value
    image[23:27, 5:20] = value

    return image


def create_icml(height, width):
    """
    Creates a 2D image that looks like a text page made of ICML letters
    Optimized version with vectorized operations
    """
    # Create dictionary of letter generators
    letters = {"I": create_i, "C": create_c, "M": create_m, "L": create_l}

    # Create output image
    result = np.zeros((height, width))

    # Page layout parameters
    top_margin = 10
    left_margin = 10
    line_spacing = 40
    word_spacing = 15
    letter_width = 25
    letter_height = 29

    # Pre-calculate all possible vertical positions
    y_positions = np.arange(
        top_margin, height - letter_height - top_margin, line_spacing
    )

    # For each line
    for y_pos in y_positions:
        # Pre-calculate possible word positions on this line
        # max_words = (width - left_margin) // (letter_width * 2 + word_spacing)
        x_positions = []
        current_x = left_margin

        # Generate words for the entire line at once
        while current_x < width - letter_width * 2:
            word_length = np.random.randint(2, 7)
            word_width = word_length * letter_width

            if current_x + word_width > width - 2:
                break

            x_positions.append((current_x, word_length))
            current_x += word_width + word_spacing

        # Generate and place all words for this line in a batch
        for x_pos, word_length in x_positions:
            # Generate word with 40% probability of "ICML" if length is 4
            if word_length == 4 and np.random.rand() < 0.4:
                word = np.hstack(
                    [
                        letters["I"](mean=1, std=0),
                        letters["C"](mean=1, std=0),
                        letters["M"](mean=1, std=0),
                        letters["L"](mean=1, std=0),
                    ]
                )
            else:
                # Generate random word
                chosen_letters = np.random.choice(
                    list(letters.keys()), size=word_length
                )
                word = np.hstack(
                    [letters[letter](mean=1, std=0) for letter in chosen_letters]
                )

            # Place word using efficient slice assignment
            result[
                int(y_pos) : int(y_pos) + letter_height, x_pos : x_pos + word.shape[1]
            ] = word

    return result


if __name__ == "__main__":
    # generate a 2D dataset using the letters as the dictionary
    np.random.seed(42)

    height = 2000
    width = 2000

    image = create_icml(height, width)

    # Save the dataset and the dictionary to a single npz file
    np.savez(
        "icml.npz",
        X=image,
        dictionary={"I": create_i(), "C": create_c(), "M": create_m(), "L": create_l()},
    )

    print("Dataset and dictionary saved to icml.npz")
