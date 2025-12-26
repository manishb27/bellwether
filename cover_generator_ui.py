import os
import io
import json
import base64
import streamlit as st
from PIL import Image, ImageOps, ImageChops, ImageFilter
from google import genai
from dotenv import load_dotenv

load_dotenv()

# Configure the client
client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

# Constants
DEFAULT_DPI = 300
BLEED_IN = 0.125
SAFE_ZONE_IN = 0.25

# PPI Constants (Pages Per Inch)
PPI_VALUES = {
    "50lb White": 440,
    "50lb Cream": 425,
    "Standard Color": 440,
    "Premium Color": 440
}

# Standard Sizes Data map
# Format: Category -> { "Label": (width, height) }
STANDARD_SIZES = {
    "Fiction (General)": {
        "4.25 x 6.87 (Mass Market)": (4.25, 6.87),
        "5 x 8 (Trade)": (5.0, 8.0),
        "5.25 x 8 (Trade)": (5.25, 8.0),
        "5.5 x 8.5 (Trade)": (5.5, 8.5),
        "6 x 9 (Trade)": (6.0, 9.0)
    },
    "Novella": {
        "5 x 8": (5.0, 8.0)
    },
    "Memoir": {
        "5.25 x 8": (5.25, 8.0),
        "5.5 x 8.5": (5.5, 8.5)
    },
    "Non-Fiction / Textbooks": {
        "5.5 x 8.5": (5.5, 8.5),
        "6 x 9": (6.0, 9.0),
        "7 x 10": (7.0, 10.0),
        "8.5 x 11": (8.5, 11.0)
    },
    "Children's Books": {
        "7.5 x 7.5 (Square)": (7.5, 7.5),
        "7 x 10": (7.0, 10.0),
        "10 x 8 (Landscape)": (10.0, 8.0)
    },
    "Photography / Art Books": {
        "8.5 x 11": (8.5, 11.0),
        "Custom": (8.5, 11.0) # Default placeholder
    }
}

# Options Data (Visuals)
GENRE_OPTIONS = [
    "Fantasy Epic book cover",
    "Sci-Fi Dystopia book cover",
    "Historical Romance book cover",
    "Psychological Thriller book cover",
    "Gothic Horror book cover",
    "Custom..."
]

FOCUS_OPTIONS = [
    "a towering ice castle crumbling under aurora lights",
    "a cybernetic assassin leaping between skyscrapers",
    "a forbidden lovers’ embrace in a moonlit garden",
    "a fractured hourglass spilling shadowy memories",
    "a cursed dollhouse with glowing red eyes in windows",
    "Custom..."
]

STYLE_OPTIONS = [
    "epic digital painting style",
    "cyberpunk vector illustration style",
    "soft watercolor realism style",
    "hyper-detailed photorealistic render style",
    "dark ink and gouache fantasy art style",
    "Custom..."
]

MOOD_OPTIONS = [
    "dramatic volumetric god rays with ethereal mist",
    "neon underglow casting long ominous shadows",
    "warm candle flicker with romantic lens flare",
    "eerie blue moonlight piercing thick fog",
    "thunderstorm flashes with lightning vein glow",
    "Custom..."
]

COMPOSITION_OPTIONS = [
    "centered heroic pose with expansive top title banner",
    "asymmetrical rule-of-thirds with right-side author panel",
    "symmetrical embrace framed by arched doorway space",
    "minimalist split-screen with floating central emblem",
    "panoramic landscape with overlaid gothic title frame",
    "Custom..."
]

COLOR_OPTIONS = [
    "icy turquoises and molten golds",
    "electric cyans against rusting industrial browns",
    "rose blush pinks with deep emerald greens",
    "desaturated grays pierced by blood crimson accents",
    "midnight indigos fading into bruised violet storms",
    "Custom..."
]

def render_option_input(label, options, default_custom_value="", key_suffix=""):
    # Default to "Custom..." if available, otherwise 0
    try:
        default_index = options.index("Custom...")
    except ValueError:
        default_index = 0

    selection = st.selectbox(label, options, index=default_index, key=f"select_{key_suffix}")
    if selection == "Custom...":
        return st.text_input(f"Custom {label}", value=default_custom_value, key=f"text_{key_suffix}")
    return selection

def generate_image_from_prompt(prompt):
    """
    Generates an image using Google's GenAI SDK (Gemini 2.5).
    """
    try:
        # Prepend instruction to ensure image generation
        full_prompt = f"Generate an image of: {prompt}"
        
        # If the prompt is a list (multimodal), we pass it directly
        # otherwise we wrap it in a list
        contents = prompt if isinstance(prompt, list) else [full_prompt]

        response = client.models.generate_content(
            model="models/gemini-3-pro-image-preview",
            contents=contents,
        )
        
        print(f"DEBUG: Response parts count: {len(response.parts) if response.parts else 0}")
        if response.parts:
            for i, part in enumerate(response.parts):
                print(f"DEBUG: Part {i} text: {part.text}")
                print(f"DEBUG: Part {i} inline_data: {part.inline_data is not None}")

        for part in response.parts:
            if part.inline_data is not None:
                # part.as_image() returns a Pydantic model wrapper, not a raw PIL Image
                # We need to convert the raw bytes to a PIL Image for Streamlit
                image_bytes = part.inline_data.data
                return Image.open(io.BytesIO(image_bytes))
            if part.text:
                 st.warning(f"Model returned text instead of image: {part.text}")
        
        return None
        return None
    except Exception as e:
        print(f"DEBUG: Exception: {e}")
        st.error(f"Generation failed: {e}")
        return None

def trim_black_borders(img, tolerance=30):
    """
    Trims black (or near-black) borders from the image.
    tolerance: 0-255 pixel value tolerence for 'black'.
    """
    if not img: return img
    
    # Convert to RGB to ensure consistent diffing
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    bg = Image.new("RGB", img.size, (0, 0, 0))
    diff = ImageChops.difference(img, bg)
    
    # Threshold: anything darker than tolerance is treated as black (0)
    # anything brighter becomes white (255)
    diff = diff.point(lambda p: 255 if p > tolerance else 0)
    
    # Get bounding box of non-black area
    bbox = diff.getbbox()
    
    if bbox:
        return img.crop(bbox)
    return img

def get_dominant_colors(img, num_colors=5):
    """Extracts dominant colors from an image."""
    img = img.copy()
    img.thumbnail((150, 150))
    # Reduce colors
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=num_colors)
    # Find dominant colors
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    
    hex_colors = []
    if color_counts:
        for count, index in color_counts[:num_colors]:
            start_idx = index * 3
            if palette and start_idx + 3 <= len(palette):
                rgb = tuple(palette[start_idx : start_idx+3])
                hex_colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
    
    return hex_colors

def update_dimensions_from_preset():
    """Callback to update session state width/height from the selected preset."""
    cat = st.session_state.get("book_category")
    size_label = st.session_state.get("size_preset")
    
    if cat and size_label and cat in STANDARD_SIZES:
        sizes = STANDARD_SIZES[cat]
        if size_label in sizes:
            width, height = sizes[size_label]
            st.session_state.trim_width = width
            st.session_state.trim_height = height

def render_cover_generator():
    st.header("Book Information")
    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Title", "My Great Novel")
        subtitle = st.text_input("Subtitle", "")
        author = st.text_input("Author", "Jane Doe")
    with col2:
        # Genre selection with custom option
        genre = render_option_input("Genre/Theme (Prompt)", GENRE_OPTIONS, "Fantasy", "genre")
        isbn = st.text_input("ISBN", "")
        blurb = st.text_area("User Input", "It should have a knight holding a sword in front of an army\n")

    st.header("Physical Specifications")
    
    # Category and Size Selection
    cat_col, size_col = st.columns(2)
    with cat_col:
        # Defaults to Fiction
        category = st.selectbox(
            "Book Category (Standard Genres)", 
            list(STANDARD_SIZES.keys()), 
            index=0, 
            key="book_category"
        )
    with size_col:
        # Populate sizes based on category
        size_options = list(STANDARD_SIZES[category].keys())
        st.selectbox(
            "Standard Size Preset", 
            size_options, 
            index=len(size_options)-1, # Default to last (usually 6x9 or larger)
            key="size_preset",
            on_change=update_dimensions_from_preset
        )

    col3, col4 = st.columns(2)
    with col3:
        # Use values from session state (controlled by preset)
        trim_width = st.session_state.trim_width
        trim_height = st.session_state.trim_height
        st.markdown(f"**Dimensions:** {trim_width}\" x {trim_height}\"")
        # target_dpi removed from UI, defaulting to 300 internally
        target_dpi = 300
        
    with col4:
        page_count = st.number_input("Page Count", value=320, step=1)
        # paper_type removed from UI, defaulting to standard white (PPI 440)
        paper_type = "Standard White"
        ppi = 440

    st.header("Design Preferences")
    col5, col6 = st.columns(2)
    with col5:
        mood = render_option_input("Mood & Lighting Description", MOOD_OPTIONS, "Atmospheric, dark, cinematic", "mood")
        style = render_option_input("Art Style & Medium Description", STYLE_OPTIONS, "Digital painting, hyper-realistic", "style")
        color_palette = render_option_input("Color Palette Description", COLOR_OPTIONS, "Dark blue and gold", "color")
    with col6:
        front_focus = render_option_input("Main Subject Focus Description", FOCUS_OPTIONS, "Front-right area for focus", "focus")
        composition = render_option_input("Composition & Layout Description", COMPOSITION_OPTIONS, "Centered, negative space at top for title", "comp")
        
        spine_constraints = st.text_input("Spine Constraints", "Low detail")
        # Explicitly enforcing no text, UI option removed
        no_text = True
    
    st.markdown("---")
    st.subheader("Reference Image (Optional)")
    uploaded_ref_img = st.file_uploader("Upload an image to guide the style/composition:", type=["png", "jpg", "jpeg"])
    reference_image = None
    if uploaded_ref_img:
        try:
             reference_image = Image.open(uploaded_ref_img)
             st.image(reference_image, caption="Reference", width=200)
             
             # influence_type removed (Defaulting to General Mix)
             
             ref_strength = st.select_slider(
                 "Image Influence Strength",
                 options=["Subtle", "Medium", "Strong", "Maximal"],
                 value="Strong",
                 help="Strong/Maximal will force the AI to prioritize the image over your text inputs."
             )
             
             # Extract colors for "Strong" or "Maximal" modes automatically
             extracted_colors = []
             if ref_strength in ["Strong", "Maximal"]:
                 extracted_colors = get_dominant_colors(reference_image)
                 st.caption(f"Extracted Palette: {', '.join(extracted_colors)}")
        except Exception:
             st.error("Invalid image file.")
    else:
        influence_type = None

    # Hardcoded Image Settings (UI options removed)
    image_resolution = "HD (High Definition)"
    image_quality = 95

    # Calculations
    # Spine Width = (Page Count / PPI)
    spine_width_in = page_count / ppi
    
    # Full Cover Dimensions
    full_width_in = (trim_width * 2) + spine_width_in + (BLEED_IN * 2)
    full_height_in = trim_height + (BLEED_IN * 2)
    
    full_width_px = int(full_width_in * target_dpi)
    full_height_px = int(full_height_in * target_dpi)
    safe_zone_px = int(SAFE_ZONE_IN * target_dpi)

    # Construct Data Object for Prompt Generation
    user_choices = {
        "title": title,
        "subtitle": subtitle,
        "author": author,
        "genre": genre,
        "mood": mood,
        "style": style,
        "composition": composition,
        "color_palette": color_palette,
        "front_focus": front_focus,
        "spine_constraints": spine_constraints,
        "dimensions": {
             "full_width_in": full_width_in,
             "full_height_in": full_height_in,
             "aspect_ratio": f"{full_width_in:.2f}:{full_height_in:.2f}"
        },
        "blurb": blurb
    }

    st.subheader("Visual Preview")
    if st.button("Generate Image"):
        with st.spinner("Refining prompt with Gemini 2.0 Flash Lite..."):
            try:
                # 1. Generate Creative Prompt
                # Variable renamed to support logic above
                # prompt_request = ( ... ) removed to avoid conflict

                
                # 1. Generate Creative Prompt
                base_instruction = (
                    f"Create a highly detailed image generation prompt for a book cover based on the following specifications:\n"
                    f"{json.dumps(user_choices, indent=2)}\n\n"
                    f"The image must be a SINGLE CONTINUOUS WIDE CINEMATIC SHOT. It will be wrapped around a book later, but the image itself must be a continuous painting.\n"
                    f"Do not include any text, letters, titles, or author names in the image (this is artwork only).\n"
                    f"Describe the scene such that the main focus is on the right and the scene extends atmospherically to the left.\n"
                    f"CRITICAL: DO NOT describe a 'spine' or 'center spine' or 'book spine'. Treat the center of the image as just the middle of the scene. DO NOT allow the model to draw a vertical line, bar, or shadow in the middle.\n"
                    f"Ensure the aspect ratio fits {full_width_in:.2f} (width) by {full_height_in:.2f} (height).\n"
                )

                refinement_contents = [base_instruction]
                
                if reference_image:
                     # General / Mix Everything Strategy
                     ref_instruction = (
                         "\n\nCRITICAL: I have attached a reference image. "
                         "Analyze the image and extract its key visual elements, characters, and setting. "
                         "The final result should look like a creative fusion of the user's requirements and this image's elements."
                     )
                         
                     base_instruction += ref_instruction
                     
                     # 1. Inject Extracted Colors if available
                     if extracted_colors:
                         base_instruction += (
                             f"\n\nCOLOR COMPLIANCE: The reference image contains the following specific color palette: {', '.join(extracted_colors)}. "
                             f"You MUST include these specific colors in the prompt descriptions to ensure visual consistency."
                         )

                     # 2. Inject Strength Instructions
                     if ref_strength == "Maximal":
                         base_instruction += (
                             "\n\nPRIORITY OVERRIDE: The Reference Image is the PRIMARY SOURCE OF TRUTH. "
                             "If the User's text description contradicts the visual style or content of the image, IGNORE the text and follow the image. "
                             "The goal is to replicate the reference image's essence almost exactly, just adapted to the book cover format."
                         )
                     elif ref_strength == "Strong":
                         base_instruction += "\n\nEnsure strong adherence to the reference image's visual identity."

                     refinement_contents = [base_instruction, reference_image]

                base_instruction += "\nReturn ONLY the prompt text."
                # Verify last item is string to append instruction logic or just rebuild list
                if isinstance(refinement_contents[-1], str):
                    refinement_contents[-1] += "\nReturn ONLY the prompt text."
                else:
                    refinement_contents.append("Return ONLY the prompt text.")

                prompt_response = client.models.generate_content(
                    model="models/gemini-2.0-flash-lite",
                    contents=refinement_contents
                )
                
                refined_prompt = prompt_response.text
                
                with st.expander("View Generated Prompt (Debug)"):
                    st.write(refined_prompt)
                
            except Exception as e:
                st.error(f"Error generating prompt: {e}")
                refined_prompt = None

        if refined_prompt:
            with st.spinner("Generating cover image..."):
                # 2. Generate Image
                
                # Append technical constraints to the refined prompt to ensure compliance
                final_prompt = (
                    f"{refined_prompt} "
                    f" --ar {full_width_in:.2f}:{full_height_in:.2f}"
                    f" NO TEXT. NO TYPOGRAPHY. NO BORDERS. NO LETTERBOXING. FILL THE WHOLE CANVAS."
                )

                if reference_image:
                     st.info("Using reference image for generation...")
                     # multimodal list: [text_prompt, image]
                     prompt_content = [final_prompt, reference_image]
                     img = generate_image_from_prompt(prompt_content)
                else:
                     img = generate_image_from_prompt(final_prompt)
                
                if img:
                    # Post-process: Trim black artifacts/letterboxing first
                    img = trim_black_borders(img)

                    # Post-process: Force alignment to calculated Aspect Ratio
                    target_w = full_width_px
                    target_h = full_height_px
                    
                    # Use PIL ImageOps.fit to center-crop/resize to exact dimensions
                    # We check scale to decide if we need to sharpen
                    curr_w, curr_h = img.size
                    scale_factor = max(target_w / curr_w, target_h / curr_h)
                    
                    img_processed = ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

                    # If we upscaled significantly (more than 10%), apply sharpening to maintain "quality" perception
                    if scale_factor > 1.1:
                         img_processed = img_processed.filter(ImageFilter.UnsharpMask(radius=2, percent=100, threshold=3))

                    # Prepare base64 for 3D view and download
                    buf = io.BytesIO()
                    img_processed.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    b64_img = base64.b64encode(byte_im).decode()
                    
                    # Optional: View Flat Spread
                    with st.expander("View Flat Print Spread (2D)"):
                         st.image(img_processed, caption="Full Spread (Back, Spine, Front)", use_container_width=True)

                    # Construct data dict for 3D view logic (needed for CSS calcs below)
                    # We reconstruct the necessary parts of the previous 'data' object
                    data = {
                        "book_info": {
                            "trim_size_in": {"width": trim_width, "height": trim_height}
                        },
                        "computed": {
                            "spine_width_in": spine_width_in,
                            "whole_cover_size_in": {"width": full_width_in, "height": full_height_in}
                        }
                    }

                    st.subheader("3D Preview")
                    
                    # Calculate dimensions for CSS
                    # We use a fixed display width for the book front to keep it responsive/uniform
                    disp_width = 220 
                    
                    # Ratios from data
                    real_trim_w = data['book_info']['trim_size_in']['width']
                    real_trim_h = data['book_info']['trim_size_in']['height']
                    real_spine_w = data['computed']['spine_width_in']
                    real_full_w = data['computed']['whole_cover_size_in']['width']
                    
                    # CSS Dimensions
                    # Height is proportional to trim height relative to trim width
                    css_h = disp_width * (real_trim_h / real_trim_w)
                    # Spine Thickness is proportional to spine width relative to trim width
                    css_thick = disp_width * (real_spine_w / real_trim_w)
                    
                    # Background Sizing
                    # We need to scale the background image so that the relevant slice fills the face.
                    
                    # Front Face: 
                    # The image width is 'real_full_w'. The front cover part is roughly 'real_trim_w'.
                    # So the background size needs to be magnified such that 'real_trim_w' equals 'disp_width'.
                    # Ratio = real_full_w / real_trim_w.
                    bg_scale_front = (real_full_w / real_trim_w) * 100
                    
                    # Spine Face:
                    # The spine part is 'real_spine_w'.
                    # Ratio = real_full_w / real_spine_w.
                    bg_scale_spine = (real_full_w / real_spine_w) * 100

                    # Calculate Background Positions for seamless wrapping
                    # Front Face: Display segment starting after (Bleed + Back + Spine)
                    # We need to shift the background LEFT by this amount.
                    # The scale is based on Trim Width (disp_width).
                    offset_front_in = BLEED_IN + real_trim_w + real_spine_w
                    # Convert inch offset to pixels relative to the display element
                    offset_front_px = (offset_front_in / real_trim_w) * disp_width
                    bg_pos_front_str = f"-{offset_front_px:.2f}px"

                    # Back Face: Display segment starting after (Bleed)
                    offset_back_in = BLEED_IN
                    offset_back_px = (offset_back_in / real_trim_w) * disp_width
                    bg_pos_back_str = f"-{offset_back_px:.2f}px"

                    # Spine Face: Display segment starting after (Bleed + Back)
                    offset_spine_in = BLEED_IN + real_trim_w
                    # Scale is based on Spine Width (css_thick)
                    offset_spine_px = (offset_spine_in / real_spine_w) * css_thick
                    bg_pos_spine_str = f"-{offset_spine_px:.2f}px"

                    # 3D Book CSS
                    # Refined to auto-rotate smoothly without Streamlit interactions.
                    
                    st.markdown(f"""
                        <style>
                        .book-viewport {{
                            perspective: 1500px;
                            display: flex;
                            justify-content: flex-start;
                            align-items: center;
                            width: 100%;
                            height: {css_h + 80}px;
                            padding-top: 40px;
                            padding-bottom: 40px;
                            padding-left: 20px;
                        }}
                        
                        /* The 3D Object Container */
                        .book-3d {{
                            position: relative;
                            width: {disp_width}px;
                            height: {css_h}px;
                            transform-style: preserve-3d;
                            animation: rotateBook 12s infinite linear;
                        }}
                        
                        /* Pause rotation on hover */
                        .book-3d:hover {{
                            animation-play-state: paused;
                        }}

                        @keyframes rotateBook {{
                            from {{ transform: rotateY(0deg) rotateX(-5deg); }}
                            to {{ transform: rotateY(360deg) rotateX(-5deg); }}
                        }}

                        .face {{
                            position: absolute;
                            background-color: #fff;
                            /* Removed border to eliminate seams */
                            /* box-shadow: inset 0 0 10px rgba(0,0,0,0.05); */ 
                        }}
                        
                        /* 
                           GEOMETRY FIX:
                           To fix the "separated spine" look, we ensure the faces meet exactly at the corners.
                           The spine is located at -X (left side).
                           Front is at +Z. Back is at -Z.
                           
                           Spine width: {css_thick}
                           Front/Back width: {disp_width}
                           
                           We move the spine to the left edge of the block.
                        */
                        
                        /* FRONT COVER (Facing +Z) */
                        /* Image slice: Right side */
                        .face.front {{
                            width: {disp_width}px;
                            height: {css_h}px;
                            /* Move it forward by half thickness */
                            transform: translateZ({css_thick / 2}px);
                            background-image: url('data:image/png;base64,{b64_img}');
                            background-size: {bg_scale_front}% 100%; 
                            background-position: {bg_pos_front_str} 0; 
                        }}
                        
                        /* BACK COVER (Facing -Z) */
                        /* Image slice: Left side */
                        .face.back {{
                            width: {disp_width}px;
                            height: {css_h}px;
                            /* Rotate 180 to face back, move it "forward" (which is backward in global space) */
                            transform: rotateY(180deg) translateZ({css_thick / 2}px);
                            background-image: url('data:image/png;base64,{b64_img}');
                            background-size: {bg_scale_front}% 100%; 
                            background-position: {bg_pos_back_str} 0;
                        }}
                        
                        /* SPINE (Facing -X -> Left) */
                        /* Image slice: Center */
                        .face.spine {{
                            width: {css_thick}px;
                            height: {css_h}px;
                            /* 
                               Rotate -90 deg to face Left.
                               Move to X=0 (Left Edge).
                               Initial Center X is css_thick/2.
                               Target is X=0. Shift = -css_thick/2.
                               Local Z is Global -X.
                               So translateZ(css_thick/2).
                            */
                            transform: rotateY(-90deg) translateZ({css_thick / 2}px);
                            background-image: url('data:image/png;base64,{b64_img}');
                            background-size: {bg_scale_spine}% 100%;
                            background-position: {bg_pos_spine_str} 0;
                        }}
                        
                        /* PAGES (Facing +X -> Right) */
                        .face.pages {{
                            width: {css_thick - 2}px; /* Slightly recessed */
                            height: {css_h - 4}px; /* Slightly shorter */
                            /* Rotate 90 deg to face Right */
                            transform: rotateY(90deg) translateZ({disp_width - (css_thick / 2)}px) translateY(2px);
                            background: repeating-linear-gradient(
                                90deg,
                                #fdfdfd 0px,
                                #e0e0e0 1px,
                                #fdfdfd 2px
                            );
                            box-shadow: inset 2px 0 5px rgba(0,0,0,0.1);
                        }}
                        
                        /* TOP (Facing -Y) */
                        .face.top {{
                            width: {disp_width}px;
                            height: {css_thick}px;
                            transform: rotateX(90deg) translateZ({css_thick / 2}px);
                            background: #fdfdfd;
                        }}

                        /* BOTTOM (Facing +Y) */
                        .face.bottom {{
                            width: {disp_width}px;
                            height: {css_thick}px;
                            transform: rotateX(-90deg) translateZ({css_h - (css_thick / 2)}px);
                            background: #fdfdfd;
                            box-shadow: 0 15px 30px rgba(0,0,0,0.4);
                        }}

                        /* Add a subtle lighting overlay to the spine to round it visually */
                        .face.spine::after {{
                            content: '';
                            position: absolute;
                            top: 0; left: 0; right: 0; bottom: 0;
                            background: linear-gradient(90deg, 
                                rgba(0,0,0,0.2) 0%, 
                                rgba(255,255,255,0.1) 30%, 
                                rgba(255,255,255,0.1) 70%, 
                                rgba(0,0,0,0.2) 100%
                            );
                        }}

                        </style>
                        
                        <div class="book-viewport">
                            <div class="book-3d">
                                <div class="face front"></div>
                                <div class="face back"></div>
                                <div class="face spine"></div>
                                <div class="face pages"></div>
                                <div class="face top"></div>
                                <div class="face bottom"></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="generated_cover.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    st.title("Book Cover Generator")
    render_cover_generator()
