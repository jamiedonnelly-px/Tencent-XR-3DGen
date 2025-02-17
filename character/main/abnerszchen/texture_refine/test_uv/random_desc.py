import os

tops_descriptions = [
    "Casual blue denim shirt", "Elegant white silk blouse", "Stylish black leather jacket", "Cozy green knit sweater",
    "Chic red peplum top", "Classic gray turtleneck", "Trendy purple off-the-shoulder blouse", "Comfy pink hoodie",
    "Sophisticated navy blue blazer", "Edgy olive green bomber jacket", "Sporty orange workout tank",
    "Versatile beige cardigan", "Bold yellow crop top", "Modern teal tunic", "Feminine lavender lace top",
    "Eye-catching gold sequin shirt", "Formal silver satin camisole", "Playful polka dot blouse",
    "Relaxed fit brown linen shirt", "Vibrant multicolored striped tee", "Breathable light blue athletic shirt",
    "Flattering coral wrap top", "Loose-fitting maroon poncho", "Tailored burgundy vest",
    "Cute pastel-colored babydoll top", "Sleek black cold-shoulder blouse", "High-neck emerald green halter top",
    "Ruffled hot pink chiffon shirt", "Vintage-inspired floral-print top", "Soft and stretchy jersey knit tee",
    "Plaid button-down shirt", "Sheer cream-colored tulle blouse", "Embroidered royal blue peasant top",
    "Fitted dark blue chambray shirt", "Asymmetrical hem turquoise tank", "Draped dark green cowl neck blouse",
    "Off-white crochet lace top", "Short-sleeved magenta polo shirt", "Velvet ruby red cami",
    "Long-sleeved charcoal grey henley", "Flowy sky blue kimono", "Business casual pinstripe shirt",
    "Layered mint green chiffon blouse", "Animal print boat-neck top", "Gingham checkered sleeveless shirt",
    "Neon-colored graphic tee", "Sparkly bronze metallic top", "Houndstooth-patterned collared blouse",
    "Distressed denim vest"
]
pants_descriptions = [
    "Casual blue jeans", "Slim-fit black trousers", "Comfortable green cargo pants", "Stylish red chinos",
    "Classic white dress pants", "Relaxed-fit gray sweatpants", "Trendy purple joggers", "Elegant brown slacks",
    "Vibrant pink capris", "Sporty orange leggings", "Chic yellow culottes", "Versatile beige khakis",
    "Cozy maroon lounge pants", "Bold teal harem pants", "Sophisticated navy blue linen trousers",
    "Edgy olive green leather pants", "Lightweight cream-colored drawstring pants",
    "Timeless charcoal grey pinstripe pants", "Fun multicolored patterned pants", "Breathable dark blue workout pants",
    "Rugged denim overalls", "Loose-fitting light blue boyfriend jeans", "Tailored burgundy wool trousers",
    "Distressed indigo skinny jeans", "High-waisted pink palazzo pants", "Flattering coral wide-leg pants",
    "Modern lavender cropped pants", "Eye-catching gold metallic leggings", "Formal silver tuxedo pants",
    "Vintage-inspired plaid pants", "Playful polka dot capris", "Durable dark green work pants",
    "Fitted tan corduroy trousers", "Athletic striped track pants", "Soft pastel-colored lounge pants",
    "Fashion-forward turquoise culottes", "Laid-back stone-washed jeans", "Smooth satin-finish pants",
    "Glamorous ruby red velvet trousers", "Feminine floral-print pants", "Comfy coffee-colored drawstring pants",
    "Sleek midnight blue cigarette pants", "Eco-friendly forest green hemp pants", "Versatile reversible pants",
    "Business casual houndstooth trousers", "Adventurous camouflage-print cargo pants",
    "High-performance moisture-wicking leggings", "Cozy fleece-lined winter pants", "Boho-chic paisley-patterned pants",
    "Sophisticated pinstripe wide-leg pants"
]
shoes_descriptions = [
    "Casual blue sneakers", "Elegant black high heels", "Stylish brown leather boots", "Comfortable green loafers",
    "Chic red sandals", "Classic white tennis shoes", "Trendy purple pumps", "Sporty orange running shoes",
    "Sophisticated navy blue oxfords", "Edgy olive green combat boots", "Colorful pink ballet flats",
    "Versatile beige wedges", "Bold yellow platform heels", "Modern teal ankle boots", "Feminine lavender kitten heels",
    "Eye-catching gold gladiator sandals", "Formal silver stilettos", "Playful polka dot espadrilles",
    "Relaxed fit brown moccasins", "Vibrant multicolored flip-flops", "Breathable light blue mesh sneakers",
    "Sleek black slip-on shoes", "High-top maroon basketball shoes", "Tailored burgundy dress shoes",
    "Rugged gray hiking boots", "Soft pastel-colored slippers", "Cozy fur-lined winter boots",
    "Strappy turquoise peep-toe heels", "Classic dark green suede loafers", "Chunky royal blue block heels",
    "Plush velvet crimson slippers", "Retro-inspired white go-go boots", "Embellished emerald green pumps",
    "Cork-soled tan platform sandals", "Braided leather dark blue thong sandals", "Studded coral gladiator sandals",
    "Pointed-toe hot pink flats", "Shiny patent leather oxblood loafers", "Lace-up charcoal grey ankle boots",
    "Breathable mesh neon-colored trainers", "Fringe-adorned beige moccasin boots", "Casual pinstripe slip-on shoes",
    "Eco-friendly forest green canvas sneakers", "Leopard print calf hair flats",
    "Rhinestone-encrusted silver evening shoes", "Sculptural heel magenta pumps",
    "Classic houndstooth-patterned loafers", "Camouflage-print lace-up boots", "Futuristic metallic rose gold sneakers"
]
suit_descriptions = [
    "Classic black two-piece suit", "Elegant navy blue pinstripe suit", "Stylish charcoal grey three-piece suit",
    "Modern slim-fit white suit", "Bold red tuxedo", "Sophisticated beige linen suit",
    "Casual olive green corduroy suit", "Trendy brown tweed suit", "Eye-catching teal velvet suit",
    "Formal silver brocade suit", "Relaxed fit light blue seersucker suit", "Chic burgundy double-breasted suit",
    "Sharp dark green windowpane suit", "Tailored houndstooth-patterned suit", "Sleek black shawl collar tuxedo",
    "Timeless grey plaid suit", "Versatile tan cotton suit", "Feminine lavender skirt suit",
    "Playful polka dot pant suit", "Vibrant multicolored checkered suit", "Breathable cream-colored tropical wool suit",
    "Bold yellow power suit", "Eco-friendly forest green hemp suit", "Casual denim suit",
    "Luxurious gold jacquard suit", "High-fashion orange silk suit", "Retro-inspired purple zoot suit",
    "Camouflage-print military-style suit", "Dapper royal blue peak lapel suit", "Sporty green tracksuit",
    "Equestrian-inspired riding suit", "Vintage plaid golf suit", "Safari-style khaki suit",
    "Floral-print garden party suit", "Sleek satin-finish suit", "Glamorous ruby red sequin suit",
    "Futuristic metallic silver suit", "Cozy fleece loungewear set", "Boho-chic paisley-patterned suit",
    "Sophisticated pinstripe skirt suit", "Crisp white linen suit", "Edgy distressed denim suit",
    "Athletic striped warm-up suit", "Elegant emerald green evening suit",
    "Nautical-inspired navy and white striped suit"
]

dname_desc_map = {
    "MCWY_2_Top": tops_descriptions,
    "MCWY_2_Bottom": pants_descriptions,
    "MCWY_2_Shoe": shoes_descriptions,
    "MCWY_2_Dress": suit_descriptions,
    "readyplayerme_Top": tops_descriptions,
    "readyplayerme_Bottom": pants_descriptions,
    "readyplayerme_Footwear": shoes_descriptions,
}
