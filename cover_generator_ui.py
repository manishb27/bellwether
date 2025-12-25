import os
import io
import json
import base64
import streamlit as st
from PIL import Image, ImageOps
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
    selection = st.selectbox(label, options, key=f"select_{key_suffix}")
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
        response = client.models.generate_content(
            model="models/gemini-3-pro-image-preview",
            contents=[full_prompt],
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
    except Exception as e:
        print(f"DEBUG: Exception: {e}")
        st.error(f"Generation failed: {e}")
        return None

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
        blurb = st.text_area("Blurb", "It should have a knight holding a sword in front of an army\n")

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
        # Linked to session state
        trim_width = st.number_input("Trim Width (in)", step=0.01, key="trim_width")
        trim_height = st.number_input("Trim Height (in)", step=0.01, key="trim_height")
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
        mood = render_option_input("Mood / Lighting", MOOD_OPTIONS, "Atmospheric, dark, cinematic", "mood")
        style = render_option_input("Art Style / Medium", STYLE_OPTIONS, "Digital painting, hyper-realistic", "style")
        color_palette = render_option_input("Color Palette", COLOR_OPTIONS, "Dark blue and gold", "color")
    with col6:
        front_focus = render_option_input("Main Visual Focus", FOCUS_OPTIONS, "Front-right area for focus", "focus")
        composition = render_option_input("Composition Details", COMPOSITION_OPTIONS, "Centered, negative space at top for title", "comp")
        
        spine_constraints = st.text_input("Spine Constraints", "Low detail")
        # Explicitly enforcing no text, UI option removed
        no_text = True

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
    if st.button("Generate Web Image (Gemini 3 Pro)"):
        with st.spinner("Refining prompt with Gemini 2.0 Flash Lite..."):
            try:
                # 1. Generate Creative Prompt
                prompt_request = (
                    f"Create a highly detailed image generation prompt for a book cover based on the following specifications:\n"
                    f"{json.dumps(user_choices, indent=2)}\n\n"
                    f"The image must be a SINGLE CONTINUOUS WIDE CINEMATIC SHOT. It will be wrapped around a book later, but the image itself must be a continuous painting.\n"
                    f"Do not include any text, letters, titles, or author names in the image (this is artwork only).\n"
                    f"Describe the scene such that the main focus is on the right and the scene extends atmospherically to the left.\n"
                    f"CRITICAL: DO NOT describe a 'spine' or 'center spine' or 'book spine'. Treat the center of the image as just the middle of the scene. DO NOT allow the model to draw a vertical line, bar, or shadow in the middle.\n"
                    f"Ensure the aspect ratio fits {full_width_in:.2f} (width) by {full_height_in:.2f} (height).\n"
                    f"Return ONLY the prompt text."
                )
                
                prompt_response = client.models.generate_content(
                    model="models/gemini-2.0-flash-lite",
                    contents=[prompt_request]
                )
                
                refined_prompt = prompt_response.text
                # st.caption("DEBUG: Refined Prompt used:")
                # st.code(refined_prompt)
                
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
                    f" NO TEXT. NO TYPOGRAPHY."
                )

                img = generate_image_from_prompt(final_prompt)
                
                if img:
                    # Post-process: Force alignment to calculated Aspect Ratio
                    target_w = full_width_px
                    target_h = full_height_px
                    
                    # Use PIL ImageOps.fit to center-crop/resize to exact dimensions
                    img_processed = ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

                    # Layout columns for 2D and 3D views
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Flat Cover (Processed)")
                        st.image(img_processed, caption="Full Spread (Back, Spine, Front)", use_container_width=True)
                    
                    # Prepare base64 for 3D view and download
                    buf = io.BytesIO()
                    img_processed.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    b64_img = base64.b64encode(byte_im).decode()
                    
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

                    with res_col2:
                        st.subheader("3D Prediction")
                    
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

                    # 3D Book CSS
                    # Refined to auto-rotate smoothly without Streamlit interactions.
                    
                    st.markdown(f"""
                        <style>
                        .book-viewport {{
                            perspective: 1500px;
                            display: flex;
                            justify-content: center;
                            align-items: center;
                            width: 100%;
                            height: {css_h + 80}px;
                            padding-top: 40px;
                            padding-bottom: 40px;
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
                            background-position: 100% 0; 
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
                            background-position: 0% 0;
                        }}
                        
                        /* SPINE (Facing -X -> Left) */
                        /* Image slice: Center */
                        .face.spine {{
                            width: {css_thick}px;
                            height: {css_h}px;
                            /* 
                               Rotate -90 deg to face Left.
                               Translate Z by half width to push it out to the edge.
                               Wait, if width is {disp_width}, half width is {disp_width/2}.
                               AND we might need a slight overlap (-0.5px) to prevent gaps.
                            */
                            transform: rotateY(-90deg) translateZ({disp_width / 2 - 0.5}px);
                            background-image: url('data:image/png;base64,{b64_img}');
                            background-size: {bg_scale_spine}% 100%;
                            background-position: 50% 0;
                        }}
                        
                        /* PAGES (Facing +X -> Right) */
                        .face.pages {{
                            width: {css_thick - 2}px; /* Slightly recessed */
                            height: {css_h - 4}px; /* Slightly shorter */
                            /* Rotate 90 deg to face Right */
                            transform: rotateY(90deg) translateZ({disp_width / 2 - 2}px) translateY(2px);
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
