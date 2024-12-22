```python
import weaviate

client = weaviate.connect_to_local()

print(client.is_ready())
```

    True
    

    D:\project\weaviate-python-client\weaviate\warnings.py:133: DeprecationWarning: Dep005: You are using weaviate-client version 0.1.dev3117+gae1bb03.d20241220. The latest version is 4.10.2.
                Consider upgrading to the latest version. See https://weaviate.io/developers/weaviate/client-libraries/python for details.
      warnings.warn(
    


```python
jeopardy = client.collections.get("OpenVid_1M")
```

## 测试任务
1、 根据描述词和相机动作召回

2、 根据描述带有窗户、食物的关键词召回

3、 带有空间标量的召回策略

4、 其它召回方式

### 矢量召回策略


```python
from weaviate.classes.query import MetadataQuery


vector_names = ["Caption", "CameraMotion"]
response = jeopardy.query.near_text(
    query="A Ice cream appears in front of the window, and the view outside the window is very peaceful.",  
    target_vector=vector_names,  # Specify the target vector for named vector collections 
    limit=2,
    return_metadata=MetadataQuery(score=True, explain_score=True),
)



for o in response.objects:
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata.score, o.metadata.explain_score)
```

    The video features an older man with white hair, wearing a blue suit and a white shirt. He is seated in front of a backdrop that displays a city skyline. The man appears to be engaged in a conversation or an interview, as he is gesturing with his hands while speaking. The style of the video suggests it could be a news segment or a talk show, given the professional attire of the man and the formal setting. The backdrop of the city skyline adds a sense of location and context to the video.
    Undetermined
    0.0 
    The video features a woman with blonde hair, wearing a red jacket, smiling and gesturing with her hands. She is seated in a coffee shop, surrounded by other patrons and a counter with various items. The setting is casual and inviting, with a warm and cozy atmosphere. The woman appears to be engaged in a conversation or sharing a story, as she smiles and makes expressive hand gestures. The video captures a moment of connection and enjoyment in a public space.
    tilt_up
    0.0 
    

### BM25关键词召回，与窗户、食物有关的文本


```python
from weaviate.classes.query import MetadataQuery

response = jeopardy.query.bm25(
    query="window food",  
    limit=3,
    query_properties=["Caption"],
    return_metadata=MetadataQuery(score=True, explain_score=True),
)



for o in response.objects:
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata.score, o.metadata.explain_score)
```

    In the video, a man is seen in a kitchen, holding a large white bowl filled with a colorful assortment of food. He is wearing a white hat and a green shirt, and he is holding a spoon in his hand. The man is smiling and appears to be enjoying the food. The kitchen has white cabinets and a window in the background. The man is standing in front of the window, which lets in natural light. The food in the bowl includes various vegetables and meat, and it looks delicious. The man seems to be preparing to eat the food, as he is holding the spoon over the bowl. The overall atmosphere of the video is warm and inviting, with the man appearing to be in a good mood.
    static
    3.759178638458252 , BM25F_window_frequency:2, BM25F_window_propLength:61, BM25F_food_frequency:4, BM25F_food_propLength:61
    In the video, a man is seen enjoying a meal at a table. He is wearing a red shirt and is seated in front of a white bowl filled with food. He is using a spoon to eat the food. The table is made of wood and is located near a window. The window has a white frame and is covered with a white curtain. The man appears to be in a good mood as he enjoys his meal. The overall atmosphere of the video is casual and relaxed.
    static
    3.5218000411987305 , BM25F_window_propLength:48, BM25F_food_frequency:2, BM25F_food_propLength:48, BM25F_window_frequency:2
    The video features a man in a kitchen, holding a plate of food. He is standing in front of a refrigerator and a window with blinds. The man is wearing a white shirt and has curly hair. He appears to be in the middle of a meal, as he is holding a plate of food. The kitchen has wooden cabinets and a stainless steel refrigerator. The window has blinds that are partially open. The man seems to be enjoying his meal, as he is holding the plate up to his face. The overall style of the video is casual and relaxed, with a focus on the man and his meal.
    static
    3.482060194015503 , BM25F_food_propLength:51, BM25F_window_frequency:2, BM25F_window_propLength:51, BM25F_food_frequency:2
    

> 看起来并没有满足需求，参杂着一些跟窗户很相似的数据，这是因为bm25的评分算法有关，这个文本中有很多关于window的形容词，它们在不同领域是不同的术语，增加BM25的b参数可以调节 https://www.cnblogs.com/novwind/p/15177871.html

### 矢量召回，增加距离排序
先筛选一个相似的范围，然后在这个空间中，按照词义进行标量排序，

这种方法适用于目标任务确定，结果符合幂等，静态数据检索等任务


```python
from weaviate.classes.query import Rerank, MetadataQuery

vector_names = ["Caption", "CameraMotion"]
response = jeopardy.query.near_text(
    query="A Ice cream appears in front of the window, and the view outside the window is very peaceful.",  
    target_vector=vector_names,  # Specify the target vector for named vector collections 
    limit=100,
    rerank=Rerank(
        prop="caption",
        query="food"
    ),
    return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
)



for o in response.objects:
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata)
```

    In the video, a woman is seen in a kitchen, preparing food. She is wearing a blue shirt and is standing in front of a counter. On the counter, there are several bowls and a cutting board. The woman is holding a piece of food in her hand and appears to be examining it. In the background, there is a potted plant and a window. The overall style of the video is casual and homey, with a focus on the woman's actions in the kitchen.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-1.7974681854248047)
    In the video, a person is preparing a meal on a wooden table. The person is using a green cutting board and a knife to slice an orange. The orange is being cut into thin slices. In the background, there is a wooden bowl filled with green leaves, possibly lettuce. The scene is set in a kitchen with a window that lets in natural light. The overall style of the video is simple and focused on the food preparation process.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.15559458732605)
    In the video, a woman is seen in a kitchen, preparing a meal. She is wearing a striped shirt and an apron, indicating her role as a chef or cook. The kitchen is well-equipped with various utensils and appliances, including a sink, a stove, and a refrigerator. The woman is seen using a can opener to open a can of tomatoes, which she then adds to a pot on the stove. She is also seen using a spoon to stir the contents of the pot. The kitchen is filled with various items, including bowls, cups, and bottles, suggesting that she is in the middle of cooking a complex meal. The overall style of the video is realistic and informative, providing viewers with a glimpse into the process of cooking.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.376378059387207)
    The video is a cooking show featuring Gwyneth Paltrow. In the image, we see a wooden table with various bowls containing different ingredients, such as salad greens, carrots, and other vegetables. There are also utensils like knives and spoons on the table. The style of the video is a typical cooking show, with a focus on the preparation of a meal. The host, Gwyneth Paltrow, is likely demonstrating a recipe or technique, and the video is likely to include close-ups of the ingredients and the cooking process. The overall atmosphere of the video is likely to be warm and inviting, with a focus on healthy and delicious food.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.403155565261841)
    The video is a collage of three images, each depicting a different scene. The first image shows a man sitting in a car, looking out of the window. The second image is a close-up of a plate of food, which includes eggs, bacon, and blueberries. The third image is a logo for Cracker Barrel, a restaurant chain. The style of the video is a montage, with each image transitioning smoothly into the next. The man in the car appears to be driving, while the food in the second image looks delicious and inviting. The Cracker Barrel logo is prominently displayed in the third image, suggesting that the video may be promoting the restaurant.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.5439376831054688)
    The video shows a man in a kitchen preparing a meal. He is standing at a counter with a wooden cutting board and a tray of meat. In the first frame, he is pouring a sauce from a bowl onto the meat. In the second frame, he is using a spoon to mix the sauce with the meat. In the third frame, he is placing the meat into a pan. The kitchen is well-equipped with various appliances and utensils. The man is wearing a black shirt and appears to be focused on his task. The style of the video is a straightforward, real-life cooking tutorial.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.5995609760284424)
    In the video, two men are seated at a round table, engaged in a lively conversation. The table is set with a variety of food and drinks, including hot dogs, bottles of beer, and cups. The men are casually dressed, with one wearing a plaid shirt and the other in a black t-shirt. They are seated on black chairs, and each has a red napkin on their lap. The background is dark, which puts the focus on the men and their interaction. The overall style of the video is casual and relaxed, capturing a moment of camaraderie between the two men.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.7520904541015625)
    In the video, a woman in a pink apron is preparing a meal in a kitchen. She is standing at a counter with various bowls and ingredients in front of her. The counter is filled with bowls containing different ingredients, such as salad greens and tortillas. The woman is using a knife to cut the tortillas into smaller pieces. The kitchen has wooden cabinets and a black countertop. In the background, there is a sink and a bowl of fruit. The video is likely a cooking tutorial, as it shows the woman following a recipe and preparing the ingredients.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-2.896550416946411)
    The video features a man in a black shirt standing in a kitchen. He is looking down, possibly at a recipe or a cooking task. The kitchen is equipped with a white refrigerator and a window with a view of the outdoors. The man appears to be focused on his task, suggesting that he might be preparing a meal or demonstrating a cooking technique. The overall style of the video is casual and informative, likely aimed at viewers interested in cooking or looking for new recipes to try.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-3.492882251739502)
    The video shows a man in a car eating a sandwich. He is wearing glasses and a light blue shirt. The car is parked on the side of a road with a white truck visible in the background. The man takes a bite of the sandwich and appears to be enjoying his meal. The video is likely a casual, everyday scene captured in a car.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-4.460694313049316)
    In the video, two men are seated at a table, engaged in a playful activity. They are both dressed in black shirts and are wearing glasses. The table is adorned with a blue bowl filled with cereal, a glass of water, and a carton of milk. The men are in the process of pouring milk into the bowl, with one man holding the carton and the other holding the bowl. The background features a white wall, decorated with a painting and a shelf. The overall atmosphere of the video is casual and fun, capturing a light-hearted moment between the two men.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-5.344922065734863)
    In the video, a woman is seen in a kitchen, preparing a drink using a blender. She is wearing a pink dress and glasses. The kitchen is equipped with a microwave, oven, and sink. On the counter, there are several cups and a bowl. The woman is pouring the drink from the blender into a glass. The scene is set in a well-lit kitchen with wooden cabinets. The woman appears to be in the process of making a smoothie or milkshake. The video captures a moment of everyday life, showcasing the woman's activity in the kitchen.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-5.826409816741943)
    The video is a scene from a TV show or movie, featuring a woman in a pink dress with a white headscarf, sitting at a table with a basket of vegetables. She is crying and appears to be upset. In the background, there is a person in a striped shirt, and a purple suitcase is visible. The style of the video is realistic, with a focus on the emotional state of the woman. The setting appears to be a kitchen or dining area, and the lighting is natural, suggesting an indoor environment. The overall tone of the video is dramatic and emotional.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-6.792113780975342)
    In the video, two women are seen in a kitchen, engaged in a conversation. The kitchen is well-lit and features a colorful backsplash. The woman on the left is wearing a blue shirt and a yellow necklace, while the woman on the right is dressed in a pink and gray sweater. They are standing near a window, which allows natural light to fill the room. A potted plant is visible in the background, adding a touch of greenery to the scene. The overall atmosphere of the video is casual and friendly, with the women appearing to enjoy their conversation.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-6.90876579284668)
    In the video, two men are seated at a table in a cozy, dimly lit room. The room is decorated with a warm, inviting ambiance, featuring a fireplace and a painting on the wall. The men are engaged in a conversation, with one man holding a cup and the other a bowl. The table is adorned with a centerpiece of pumpkins and gourds, adding a touch of autumn to the scene. The men are dressed casually, with one wearing a green t-shirt and the other a black shirt. The overall style of the video is realistic, capturing the essence of a casual, intimate conversation between two friends in a comfortable setting.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-6.944281578063965)
    The video features a young woman with glasses, wearing a plaid shirt. She is seen in a restaurant setting, with a dining table and chairs visible in the background. The woman is captured in three different frames, each showing her in a different pose. In the first frame, she is seen looking directly at the camera, her mouth slightly open as if she is about to speak. In the second frame, she is seen with her mouth closed, her gaze directed off to the side. In the third frame, she is seen with her mouth open, as if she is in the middle of speaking. The video captures the woman's expressions and movements, providing a glimpse into her personality and the setting in which she is in.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-7.197833061218262)
    In the video, a man and a young boy are standing in a room with a painting on the wall. The man is wearing an orange vest and a blue shirt, while the boy is wearing glasses. They are both looking at each other. The room has a staircase in the background. The man is standing in front of the boy. The boy is standing in front of the painting. The man is wearing a blue shirt. The boy is wearing a plaid shirt. The man is wearing glasses. The boy is wearing a tie. The man is wearing a vest. The boy is wearing a shirt. The man is wearing a tie. The boy is wearing a vest. The man is wearing a shirt. The boy is wearing a tie. The man is wearing a vest. The boy is wearing a shirt. The man is wearing a tie. The boy is wearing a vest. The man is wearing a shirt. The boy is wearing a tie. The man is wearing a vest. The boy is wearing a shirt. The man is wearing a tie. The boy is wearing a vest. The man is wearing a shirt. The boy is wearing a tie. The man is wearing a vest. The boy is wearing a shirt. The man is wearing a tie. The boy is wearing a vest. The man is wearing a shirt. The boy is wearing a tie. The man is wearing a vest. The boy is wearing a shirt. The man is wearing a tie. The boy
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-8.566661834716797)
    The video is a breathtaking aerial view of a tropical island, showcasing its natural beauty and vibrant marine life. The island is surrounded by a stunning turquoise lagoon, with a white sandy beach that gently slopes into the water. The lagoon is teeming with colorful coral reefs, creating a vibrant underwater landscape. The island itself is lush and green, with a dense canopy of trees and vegetation. In the distance, the horizon is dotted with other islands, adding depth and scale to the scene. The video is shot from a high angle, providing a comprehensive view of the island and its surroundings. The style of the video is realistic, capturing the natural beauty of the island and its marine life in stunning detail. The colors are vibrant and the lighting is bright, highlighting the island's natural beauty. The video is a perfect representation of a tropical paradise, showcasing the island's unique features and the stunning marine life that calls it home.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-8.880730628967285)
    In the video, a man in a blue hoodie and orange pants is seen in a wooded area, using an axe to chop a log. The log is lying on the ground, and the man is standing over it, focusing on his task. The scene is set in a natural environment, with trees and bushes surrounding the man. The man's actions suggest that he is preparing the wood for a fire or some other purpose. The video captures the man's skill and determination as he works on the log.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-8.88930606842041)
    The video is a news segment featuring a panel of six people, three men and three women, sitting around a coffee table in a studio setting. The panel is engaged in a discussion, with one man standing and speaking while the others listen attentively. The studio is decorated with Christmas trees and a potted plant, and there are books and a coffee cup on the table. The news ticker at the bottom of the screen displays headlines such as "Jack Pack's Back" and "Britain's Got Talent". The overall style of the video is professional and polished, with a focus on the panel's discussion and the news headlines.
    zoom_in+pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.022051811218262)
    The video shows a young man sitting in a booth at a diner, using his phone. He is wearing a red shirt and has a smile on his face. The booth is red and black, and there is a painting of a woman on the wall behind him. The man is holding the phone in his hands, and it appears that he is either texting or browsing the internet. The overall atmosphere of the video is casual and relaxed, with the man enjoying his time at the diner.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.103610038757324)
    The video features a man wearing a straw hat and glasses, sitting in a garden with a brick wall in the background. The man is dressed in a plaid shirt and appears to be in a relaxed state. The garden is filled with various plants and flowers, creating a peaceful and serene atmosphere. The man's gaze is directed off to the side, suggesting he is lost in thought or observing something in the distance. The overall style of the video is casual and laid-back, capturing a moment of tranquility in a natural setting.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.163442611694336)
    The video is a 3D animated scene featuring two characters. The main character is a young girl with a red hijab, who is reading a book. She is holding the book with both hands and appears to be engrossed in the story. The book has a colorful cover with the title "Nabi Daud" visible. The girl's expression is one of surprise or excitement, as she is covering her mouth with her hand.  In the background, there is an adult woman, presumably the girl's mother, who is smiling and looking at the girl with a sense of pride or affection. The woman is wearing a blue hijab and a pink top. The setting appears to be a cozy room with a window and a chair visible in the background. The overall style of the animation is bright and colorful, with a focus on the characters and their expressions. The animation is likely aimed at a younger audience, given the child-friendly design and the educational content of the book being read.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.183140754699707)
    The video shows two young men sitting on a couch, engaged in a lively conversation. They are both smiling and appear to be enjoying each other's company. The man on the left is wearing a blue shirt and a baseball cap, while the man on the right is wearing a gray t-shirt. The couch they are sitting on is brown and has a few pillows on it. In the background, there is a lamp and a window with blinds. The overall atmosphere of the video is casual and friendly.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.210822105407715)
    The video features a man in a black hat and sunglasses, standing in a park with lush greenery and colorful flowers. He is wearing a black jacket and a gray shirt, and he has a lanyard around his neck. The man appears to be speaking, and his expression is one of surprise or amusement. The park is filled with trees and plants, and there are a few other people visible in the background. The overall style of the video is casual and relaxed, capturing a moment of leisure in a beautiful outdoor setting.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.232440948486328)
    In the video, three men are standing in a snowy field, engaged in a conversation. The man on the left is wearing a red and blue jacket, the man in the middle is dressed in a camouflage jacket, and the man on the right is wearing a brown jacket with a fur-lined hood. They are standing close to each other, indicating a sense of camaraderie or shared purpose. The snow-covered ground and the trees in the background suggest a cold, winter setting. The men appear to be focused on their conversation, indicating that the content of their discussion is important to them. The overall style of the video is realistic, capturing a moment in the lives of these three men in a natural, outdoor setting.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.300634384155273)
    The video shows a man in a black shirt holding a rifle. He is wearing a headset and glasses. The man is standing in front of a building. The rifle has a scope on it. The man is pointing the rifle towards the camera. The man is speaking into the headset. The man is holding the rifle with both hands. The man is looking at the camera. The man is wearing a black shirt. The man is wearing a headset. The man is wearing glasses. The man is holding a rifle. The man is standing in front of a building. The man is pointing the rifle towards the camera. The man is speaking into the headset. The man is holding the rifle with both hands. The man is looking at the camera. The man is wearing a black shirt. The man is wearing a headset. The man is wearing glasses. The man is holding a rifle. The man is standing in front of a building. The man is pointing the rifle towards the camera. The man is speaking into the headset. The man is holding the rifle with both hands. The man is looking at the camera. The man is wearing a black shirt. The man is wearing a headset. The man is wearing glasses. The man is holding a rifle. The man is standing in front of a building. The man is pointing the rifle towards the camera. The man is speaking into the headset. The man is holding the rifle with both hands. The man is looking at the camera. The man
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.391399383544922)
    The video features a man with a beard and glasses, wearing a black suit and a white shirt with a floral pattern. He is holding a microphone and appears to be speaking or singing. The setting includes a potted plant with pink and white flowers, and a decorative vase with a geometric pattern. The man is seated in front of a backdrop with a warm orange hue. The video has a news or interview style, with a banner at the bottom that reads "CORTE & CONFESION" and a subtitle that says "Me duelle mucho que Fabian se vaya." The overall atmosphere of the video is formal and professional.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.420918464660645)
    In the video, a woman stands in a spacious, well-lit attic bedroom. She is dressed in a black tank top and has her hair styled in a braid. Her arms are outstretched, and she appears to be speaking or gesturing. The room is furnished with a large, comfortable bed and a dining table with chairs. The walls are painted white, and there are several windows that let in natural light. The overall style of the video is casual and relaxed, capturing a moment of everyday life in a beautiful, serene setting.
    zoom_in+pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.539199829101562)
    The video features a bald man with a serious expression, standing in front of a lush green tree. He is wearing a dark blue shirt and appears to be speaking or listening intently. The man is the main subject of the video, and his position in front of the tree suggests that he is outdoors. The tree provides a natural backdrop to the man, and its vibrant green leaves contrast with his dark shirt. The overall style of the video is straightforward and focused on the man, with no additional elements or distractions. The video seems to be a close-up shot, as the man fills most of the frame. The lighting in the video is natural, suggesting that it was filmed during the day. The man's serious expression and the outdoor setting give the video a professional and serious tone.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.54731559753418)
    The video is an aerial view of a traditional Chinese temple complex. The temple is surrounded by lush green trees and a large open courtyard. The temple itself has a distinctive red roof with intricate designs and multiple eaves. The courtyard is paved with red bricks and features a large stone statue in the center. The temple complex is nestled in a serene and peaceful environment, with the surrounding trees providing a natural boundary. The video captures the beauty and tranquility of the temple complex, showcasing its traditional architecture and the harmony between man-made structures and nature.
    zoom_out
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.582879066467285)
    The video captures the breathtaking view of a rocky cliff overlooking a turquoise ocean. The cliff is adorned with lush greenery and palm trees, adding a touch of tropical charm to the scene. The ocean, a stunning shade of turquoise, stretches out into the horizon, meeting the clear blue sky at the edges. The sun casts a warm glow on the scene, highlighting the vibrant colors of the ocean and the cliff. The video is taken from a high vantage point, providing a panoramic view of the ocean and the cliff. The overall style of the video is serene and picturesque, capturing the natural beauty of the location.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.604816436767578)
    The video shows a man with a beard and glasses, wearing a blue hoodie and a black baseball cap, sitting at a table. He is holding a black pen and appears to be drawing or writing. The table is white, and there is a colorful, abstract design on it that seems to be created by the man with the pen. The design is made up of red, orange, and blue lines and shapes, and it appears to be in motion, as if the man is creating it in real-time. The style of the video is casual and artistic, with a focus on the man's creative process and the vibrant colors of the design.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.61288070678711)
    The video features three animated animal characters standing in a grassy area with a building in the background. The first character is a gray cat with a pink collar and a bell around its neck. The second character is a brown dog wearing a red cap with a white bone on it. The third character is a tan dog with a pink collar and a purple bow on its head. The characters are standing close to each other, and they appear to be looking at something off-screen. The style of the animation is colorful and cartoonish, with a focus on the characters' expressions and poses. The background is simple and does not distract from the characters.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.632399559020996)
    In the video, a man and a woman are standing in front of a large, white recreational vehicle (RV). The man is gesturing with his hands, possibly explaining something to the woman. The RV is parked on a gravel lot, and there are steps leading up to the entrance. The woman is wearing a striped shirt and jeans, while the man is dressed in a black jacket and pants. The RV has a large window on the side, and there is a logo on the side that reads "TRANSCEND XPOLR." The overall style of the video is casual and informative, with the man and woman appearing to be engaged in a conversation about the RV.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.651652336120605)
    The video captures a thrilling off-road adventure in a rocky forest. A man in a black t-shirt with a skull design on the back and a white baseball cap is seen walking towards a white Jeep Wrangler. The Jeep is parked on a rocky path, surrounded by large boulders and trees. The man is carrying a green water bottle in his back pocket. The Jeep is equipped with a spare tire mounted on the back and a black roof rack. The scene is set in a forest with a mix of sunlight and shadows, creating a dynamic and adventurous atmosphere. The man's casual attire and the Jeep's rugged design suggest an outdoor adventure in a challenging terrain.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.659052848815918)
    The video features a man in a camouflage baseball cap and a brown t-shirt, standing in a wooded area with trees and foliage. He appears to be in a relaxed state, possibly enjoying the outdoors. The man is the main subject of the video, and there are no other significant objects or people in the scene. The style of the video is casual and naturalistic, capturing a moment of leisure in a serene, wooded setting.
    tilt_down
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.665687561035156)
    The video features a man in a blue t-shirt with the words "MY CAKE" on it, standing in front of a red background. He appears to be speaking or gesturing with his hands. In the background, there is a split screen showing two women. The woman on the left is wearing a black and white striped top, while the woman on the right is wearing a red and black top. The man seems to be addressing the audience, possibly discussing the women or the content of the split screen. The overall style of the video is casual and informal, with a focus on the man's speech or gesture.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.698225975036621)
    The video shows a young boy with glasses, wearing a blue shirt with a red collar and a striped sweater vest. He is seated in a chair, looking down at a book or a piece of paper, which he is holding in his hands. The boy appears to be focused on his task, possibly reading or writing. The setting is indoors, with a blurred background that suggests a home or a school environment. The lighting is soft and natural, indicating that the scene takes place during the day. The style of the video is realistic and candid, capturing a quiet moment of the boy's daily life.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.714247703552246)
    The video is a playful and imaginative scene featuring two toy vehicles, a red race car and a brown tow truck, set in a grassy field. The red race car, with its vibrant color and lightning bolt design, is the main focus of the video. It is positioned in the foreground, appearing larger and more detailed than the tow truck. The tow truck, with its brown color and smaller size, is positioned in the background, adding depth to the scene. The grassy field provides a natural and contrasting backdrop to the toy vehicles, enhancing their colors and details. The overall style of the video is whimsical and fun, capturing the essence of childhood play and imagination.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.724800109863281)
    In the video, a man in a green suit and a woman in a yellow dress are engaged in a conversation. The man is wearing glasses and a tie, while the woman is holding a drink in her hand. They are standing in a room with a green wall and a chandelier hanging from the ceiling. In the background, there is another woman who is also holding a drink. The overall style of the video is formal and elegant, with a focus on the interaction between the man and the woman in the foreground.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.743074417114258)
    The video shows a person lighting a fire in a fire pit. The fire pit is filled with wooden planks and is surrounded by rocks. The person is using a lighter to ignite the fire. The fire starts to burn brightly, with flames visible on the wooden planks. The fire pit is located outdoors, and the person is standing over it. The person is wearing a hat and is holding the lighter in their hand. The fire pit is made of metal and has a black grate on top. The fire is the main focus of the video, and the person is actively involved in starting it. The video captures the process of starting a fire in a fire pit, from the initial lighting to the flames burning brightly.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.754475593566895)
    The video shows a man in a blue and white Hawaiian shirt, wearing glasses, engaged in a conversation with another man in a grey shirt. They are in a workshop setting with various tools and equipment around them. The man in the Hawaiian shirt is looking down, possibly at a piece of paper or a device, while the man in the grey shirt is looking up, possibly at the man in the Hawaiian shirt or something else in the room. The workshop appears to be well-equipped, with a drill press, a workbench, and various other tools and materials scattered around. The lighting in the room is bright, suggesting it is daytime. The style of the video is candid and informal, capturing a moment of interaction between the two men in a real-life setting.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.758919715881348)
    In the video, a man and a woman are sitting in folding chairs on a cliff overlooking a vast desert landscape. The man is wearing a cap and glasses, while the woman has her hair tied back. They are both looking out at the expansive view, which includes a large canyon and a clear blue sky. In the background, a truck is parked on the side of the road, suggesting that they have driven there. The overall style of the video is a casual, outdoor adventure, capturing the beauty of nature and the sense of freedom that comes with exploring new places.
    pan_left
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.774116516113281)
    The video shows a young boy with glasses sitting at a table, smiling and looking at the camera. He is wearing a blue sweater with a red collar and a white shirt underneath. The boy has blonde hair and is wearing glasses. In the background, there is another person who is blurred out, suggesting that the focus is on the boy. The setting appears to be an indoor room with a plant in the background. The style of the video is casual and candid, capturing a moment of the boy's life.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.8204927444458)
    The video is a promotional advertisement for a red sports car. The style of the video is sleek and modern, with a focus on the car's design and performance. The car is shown in three different frames, each highlighting a different aspect of the vehicle. In the first frame, the car is shown from a side angle, emphasizing its aerodynamic shape and sleek lines. In the second frame, the car is shown from a front angle, showcasing its powerful engine and aggressive stance. In the third frame, the car is shown from a rear angle, highlighting its sporty design and stylish taillights. The car is parked on a road with a clear blue sky in the background, adding to the overall appeal of the vehicle. The man standing next to the car is dressed in a white shirt and jeans, giving a casual yet stylish vibe to the advertisement. The overall style of the video is polished and professional, with a focus on the car's design and performance.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.820854187011719)
    In the video, a woman is seen relaxing in a luxurious living room. She is dressed in a white robe and is seated on a comfortable couch. The couch is adorned with patterned pillows, adding to the room's opulence. The woman is being attended to by a person who is kneeling on the floor, providing her with a foot massage. The room is well-lit, with natural light streaming in from a large window. The overall atmosphere of the video is one of relaxation and indulgence.
    tilt_down+pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.824710845947266)
    The video is a vibrant and colorful animation set against a purple background. It features geometric shapes, including triangles and rectangles, in various sizes and orientations. The shapes are filled with different colors, such as blue, pink, and yellow, and they are scattered throughout the video. The animation also includes a black circle and a white square, which are positioned in the center of the video. The overall style of the video is reminiscent of retro or 80s graphics, with a playful and abstract design. The video does not contain any text or depict any specific actions or movements.
    tilt_up
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.830892562866211)
    In the video, a man and a woman are engaged in a conversation. The man is wearing a blue shirt and is seated, while the woman, with her long red hair, is standing. They are both smiling, indicating a friendly and positive interaction. The setting appears to be an indoor space with a window in the background, suggesting a casual and relaxed atmosphere. The video captures a moment of connection between the two individuals, with their body language and expressions conveying a sense of warmth and camaraderie.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.832585334777832)
    The video is a dramatic scene from a movie featuring a group of soldiers. The soldiers are dressed in military uniforms and helmets, indicating a setting of war or conflict. The soldiers are standing in a line, facing each other, suggesting a moment of confrontation or discussion. The focus is on two soldiers in the foreground, who are engaged in a face-to-face interaction, possibly a conversation or a confrontation. The background shows more soldiers, indicating that this is a larger group. The lighting and composition of the scene suggest a tense and serious atmosphere. The style of the video is realistic, with attention to detail in the costumes and setting, and a focus on the emotional and dramatic aspects of the scene.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.83785629272461)
    The video features a man with a beard and dreadlocks, wearing a black leather vest and a black wristband. He is holding a microphone and appears to be speaking or singing. The man is gesturing with his right hand, pointing upwards towards the ceiling. The background is dark, with a blurred image that suggests an indoor setting, possibly a stage or a concert venue. The style of the video is a close-up shot of the man, focusing on his facial expressions and gestures, with the microphone and his hand as the main objects in the frame. The lighting is dim, highlighting the man's features and the microphone. The overall mood of the video is intense and dramatic, with the man's expressive gestures and the microphone suggesting a performance or a speech.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.838923454284668)
    The video captures a thrilling scene of a motorcycle rider navigating a winding dirt road in a mountainous desert landscape. The rider, clad in protective gear, is seen leaning into a turn, kicking up a cloud of dust behind them. The motorcycle, a sleek black model, stands out against the earthy tones of the terrain. The road itself is a narrow dirt path, curving sharply around the rocky cliffs that flank it. The sky above is a clear blue, with a few clouds scattered across it, adding to the sense of vastness and isolation. The perspective of the video is from the viewpoint of a car following the motorcycle, giving a sense of the speed and agility of the rider. The overall style of the video is dynamic and adventurous, capturing the thrill of off-road motorcycling in a stunning natural setting.
    pan_left
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.851574897766113)
    The video features a woman with red hair, smiling and looking directly at the camera. She is wearing a grey sweater and appears to be outdoors, possibly in a desert or arid environment, as suggested by the sandy ground and sparse vegetation in the background. The lighting in the video is natural, with the sun casting shadows on the woman's face and the ground. The style of the video is candid and informal, capturing a moment of the woman's life in a natural setting.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.8787260055542)
    The video is a television show featuring two men in suits sitting on a couch. The man on the right is gesturing with his hand while speaking, and there is a book on the table in front of him. The man on the left is holding a piece of paper. The background is a cityscape with lit buildings, suggesting an urban setting. The style of the video is a talk show or interview, with the two men engaged in a conversation.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.885242462158203)
    The video shows a group of four men sitting in a casual setting, possibly a living room or a casual meeting space. They are engaged in a conversation, with one man holding a remote control, suggesting they might be watching something or about to play a video game. The men are dressed in casual attire, with one man wearing a blue shirt and the others in black or grey shirts. The room has a brick wall in the background, and there is a colorful abstract painting on the wall. The overall style of the video is informal and relaxed, capturing a moment of social interaction among friends or colleagues.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.895434379577637)
    The video is a drone shot of a winding road through a mountainous landscape. The road is surrounded by lush greenery and a river runs parallel to it. The colors in the video are vibrant, with the green of the trees and the blue of the river contrasting against the gray of the road. The perspective of the drone shot gives a sense of the vastness of the landscape and the winding nature of the road. The overall style of the video is naturalistic, capturing the beauty of the landscape without any human intervention.
    zoom_in+tilt_down
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.906390190124512)
    The video captures the breathtaking view of a natural rock arch formation in the ocean. The arch, composed of gray rock, stands majestically against the backdrop of the clear blue sky. The water surrounding the arch is a vibrant shade of blue, reflecting the sunlight and creating a mesmerizing spectacle. The arch is situated in the middle of the ocean, with the vast expanse of the sea visible on either side. The perspective of the video is from a distance, allowing the viewer to appreciate the grandeur of the arch and its surroundings. The video is a testament to the beauty of nature and the awe-inspiring power of the ocean.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.910689353942871)
    The video captures a group of people, both men and women, sitting in a large room. The room is filled with people, all dressed in red robes, suggesting a uniformity in their attire. The people are seated in rows, facing the same direction, indicating a sense of order and discipline. The room is filled with a sense of quiet and stillness, as the people appear to be in a state of meditation or prayer. The video is shot from a low angle, looking up at the people, which adds a sense of depth and perspective to the scene. The overall style of the video is one of solemnity and reverence, capturing a moment of collective spiritual practice.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.920737266540527)
    The video captures a moment of triumph for a hockey player, who is the main subject of the video. He is seen wearing a black and white cap with the words "Stanley Cup Champions 2016" emblazoned on it, signifying his team's recent victory. His face is adorned with a beard, adding to his rugged appearance. He is dressed in a white jersey, which is typical attire for a hockey player.  The setting of the video is a bustling hockey rink, filled with other players and spectators. The rink is a hive of activity, with people moving about and engaging in various activities. The atmosphere is one of excitement and celebration, as the player's team has just won the Stanley Cup.  The style of the video is dynamic and energetic, capturing the essence of the sport and the emotions of the moment. The camera angles and movements are varied, providing a comprehensive view of the scene. The focus is on the player, but the background activity adds depth and context to the story. The video is a snapshot of a moment of victory, capturing the joy and pride of the player and his team.
    pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.931924819946289)
    The video captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway. It shows a drone view of waves crashing against the rugged cliffs along Big Sur's Garay Point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.
    zoom_in+tilt_up
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.952908515930176)
    The video captures the majestic Eurasia Bank building, a towering skyscraper that stands as a symbol of modern architecture. The building's facade is a striking combination of brown and green, reflecting the sunlight and creating a mesmerizing play of colors. The building's unique design features a pointed top, adding to its grandeur. The video is taken from a low angle, emphasizing the height and scale of the building. The sky in the background is a clear blue, dotted with a few clouds, providing a serene backdrop to the urban landscape. The overall style of the video is a blend of architectural beauty and natural tranquility, showcasing the Eurasia Bank building as a marvel of modern design.
    pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.96112060546875)
    The video depicts a large, gray tank traversing a sandy desert landscape. The tank is equipped with a large gun barrel on top, suggesting it is a military vehicle designed for combat. The tank is moving across the sandy terrain, leaving tracks behind it. In the background, there are rocky formations and a large hill, indicating a rugged and arid environment. The lighting suggests it is daytime, and the sky is clear. The style of the video is realistic, with attention to detail in the tank's design and the surrounding environment. The video captures the tank's movement and the vastness of the desert landscape.
    pan_left
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.973309516906738)
    The video shows a young boy with glasses, wearing a blue sweater with a red collar and a white stripe on the sleeve. He is standing in a room with a white wall and a large screen in the background. The boy is looking to the side with a slight smile on his face. The room appears to be a classroom or a study area, and there is a plant in the corner. The style of the video is a simple, candid shot, capturing a moment in the boy's day.
    tilt_up
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.978726387023926)
    The video shows a man wearing a black mask covering his mouth and nose. He is standing in front of a chain-link fence and a brick wall. The man is gesturing with his right hand, possibly speaking or explaining something. The background is blurred, but it appears to be an outdoor setting with other people present. The style of the video is candid and informal, capturing a moment in the man's day.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-9.997721672058105)
    The video features a man in a black jacket and a black hat, standing against a white background. He appears to be speaking, as suggested by his open mouth and expressive eyes. The man is the only object in the video, and there is no text or additional elements present. The style of the video is simple and straightforward, focusing solely on the man and his expression. The white background provides a stark contrast to the man's dark attire, making him the central focus of the video. The video does not contain any action or movement, and the man's position remains constant throughout the frames. The overall impression is one of a still image or a paused video, with no indication of movement or progression.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.037353515625)
    In the video, a woman with long blonde hair is seen in a close-up shot. She is wearing a black top and appears to be engaged in a conversation with a man who is not visible in the frame. The woman's expression is serious, and she seems to be listening intently to the man. The setting appears to be indoors, with a brick wall visible in the background. The lighting is warm and soft, suggesting an intimate and relaxed atmosphere. The woman's hair is styled in loose waves, and her makeup is natural, enhancing her overall appearance. The video captures a moment of quiet interaction between the two individuals, with the woman's focused gaze and the man's unseen presence creating a sense of connection and engagement.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.051982879638672)
    In the video, a man with blue eyes and gray hair is engaged in a conversation with a woman. The man is wearing a blue shirt with a polka dot pattern and has a slight beard. The woman is wearing a floral dress and has blonde hair. They are standing in front of a blue wall with a green door. The man is speaking and looking at the woman, while the woman is listening attentively. The overall style of the video is casual and intimate, capturing a moment of interaction between the two characters.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.073946952819824)
    The video features a man in a blue polo shirt with a red logo on the left chest. He is seated in an office setting with a wooden door and a framed picture on the wall behind him. The man appears to be speaking, as indicated by his open mouth and engaged expression. The lighting in the room is soft and even, suggesting an indoor environment. The style of the video is straightforward and documentary-like, focusing on the man and his immediate surroundings without any additional embellishments or distractions.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.076945304870605)
    The video features a young man with glasses, wearing a gray t-shirt, standing in a room with a white ceiling. He is looking directly at the camera with a slight smile on his face. The room is decorated with string lights that are turned on, creating a warm and inviting atmosphere. The man appears to be in a good mood and is engaging with the viewer. The style of the video is casual and personal, suggesting that it might be a vlog or a personal video. The focus is on the man and his expression, with the string lights adding a touch of ambiance to the scene.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.102651596069336)
    The video shows a close-up of a black car with a yellow and green emblem on its hood. The emblem features a prancing horse, which is a well-known symbol of a luxury car brand. The car's hood is shiny and reflects the surrounding environment, including trees and a red object. A person's hand is visible in the bottom right corner of the frame, pointing towards the emblem. The style of the video is realistic and it captures the details of the car and its emblem in a clear and focused manner.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.11436653137207)
    The video shows a gray Hyundai car parked on a street. The car is positioned in the center of the frame, facing the camera. The car has a sleek design with a large grille and sharp headlights. The car's body is shiny, reflecting the sunlight. The street is made of asphalt and is surrounded by grass on both sides. In the background, there are mountains and a clear blue sky. The car appears to be stationary, and there are no people or other vehicles visible in the video. The style of the video is a straightforward, clear shot of the car, with no additional action or movement. The focus is solely on the car and its surroundings.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.121560096740723)
    In the video, a man with a beard and a hat is seen playing a guitar in a guitar shop. He is holding the guitar and appears to be in the middle of a conversation with another man who is standing next to him. The guitar shop is filled with various guitars hanging on the wall, creating a colorful and vibrant backdrop. The man playing the guitar seems to be explaining something to the other man, possibly about the guitar he is holding or the shop itself. The overall atmosphere of the video is casual and friendly, with the two men engaging in a conversation amidst the array of guitars.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.135639190673828)
    The video is an aerial view of a large sports complex. The complex features a football field with a track around it, a large parking lot, and several buildings. The football field is green with white lines marking the field. The track is black with white lines marking the lanes. The parking lot is filled with cars and trucks. The buildings are large and appear to be made of brick. The complex is surrounded by trees and a river. The sky is cloudy and the lighting is overcast. The style of the video is realistic and it captures the details of the complex and its surroundings.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.147732734680176)
    The video shows a man standing in front of a building with a brick facade. He is wearing a black hoodie and blue jeans. In the first frame, he is looking directly at the camera with his arms crossed. In the second frame, he has turned his head to the side, still with his arms crossed. In the third frame, he has turned his head back to the camera, but his arms are now at his sides. The man appears to be in a relaxed posture, and the building behind him provides a contrasting backdrop to his casual attire. The video seems to capture a candid moment of the man, possibly in an urban setting.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.153203010559082)
    The video features a man speaking in front of a backdrop with the Texas A&M logo. He is wearing a black shirt and has a beard. The man appears to be in the middle of a conversation or presentation, as he is looking to the side and has his mouth open as if he is speaking. The style of the video is a standard interview or presentation style, with the subject in focus and the background providing context. The lighting is bright and even, suggesting an indoor setting. The man's expression is serious, indicating that the topic of discussion is likely important or serious in nature.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.174407958984375)
    The video depicts a classroom scene with three students. The first student, a young man, is wearing glasses and a tie, and he is speaking to the camera. The second student, a young woman, is sitting at her desk, looking down at her work. The third student, another young man, is also sitting at his desk, writing on a piece of paper. The classroom is filled with books and other educational materials, and there are colorful posters on the walls. The students are all focused on their work, and the atmosphere is one of concentration and learning.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.185630798339844)
    The video is a 3D animation that showcases a human head with a glowing brain. The style is realistic with a focus on the human brain, which is depicted in a detailed and anatomically accurate manner. The brain is highlighted with a warm, orange glow, suggesting activity or function. The head is shown in profile, facing to the right, with the brain occupying the top half of the image. The background is a dark blue gradient, which contrasts with the glowing brain and adds depth to the image. The overall effect is a visual representation of the human brain in action, emphasizing its complexity and importance.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.192827224731445)
    The video features a man in a professional setting, likely a business or corporate environment. He is dressed in a brown suit jacket and a blue shirt, with a white pocket square adding a touch of elegance. His hands are clasped together, suggesting he is in the middle of a conversation or presentation. The background is a plain white, which puts the focus entirely on him. The overall style of the video is professional and polished, with a clear emphasis on the man and his attire.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.194173812866211)
    The video features a young man in a casual outdoor setting. He is wearing a plaid shirt, a baseball cap, and sunglasses. The sunglasses are large and have a reflective surface. The man is looking directly at the camera with a neutral expression. The background shows a lush green field with trees and a clear sky. The overall style of the video is casual and relaxed, with a focus on the man's attire and the natural surroundings.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.20152759552002)
    The video features a man in a suit and tie, standing in front of a microphone, speaking to an audience. He appears to be in a formal setting, possibly a conference or a press event. The man is the main subject of the video, and he is captured in three different frames, each showing him in a different pose or expression. The style of the video is straightforward and professional, focusing on the man and his speech. The background is blurred, drawing attention to the man and his message. The lighting is bright, highlighting the man and making him the focal point of the video. The video does not contain any other significant objects or elements.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.274543762207031)
    The video captures a candid moment inside a car, featuring two men engaged in a conversation. The man in the passenger seat, wearing a green jacket, is seen driving the car, while the man in the backseat, dressed in a blue t-shirt, is holding a camera and appears to be recording the interaction. The car is parked on a street lined with buildings, and a bus can be seen in the background. The overall style of the video suggests a casual, unscripted moment, possibly a vlog or a personal video diary.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.309037208557129)
    The video shows a young man sitting in the bed of a silver pickup truck. He is wearing a purple t-shirt and blue shorts. The truck is parked in a parking lot with other cars and buildings in the background. The man appears to be looking at the camera with a slight smile on his face. The style of the video is casual and candid, capturing a moment in the man's day.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.349465370178223)
    The video is a screenshot from a YouTube video, featuring a man in a black hoodie standing in front of a stone wall. The man has a beard and is looking directly at the camera with his arms outstretched. The video has 215,196 views and is titled "UKIP Needs You". The channel name is "Count Dankula" and the video has 409k subscribers. The overall style of the video is a straightforward, unfiltered representation of the man and his message.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.350701332092285)
    The video is a news segment featuring a man named Anderson, who is a student. He is seen riding a bicycle on a city street. The street is busy with traffic, including cars and a white van. The man is wearing a black and white shirt, a cap, and glasses. The background shows tall buildings, indicating an urban setting. The style of the video is informative, with a focus on the man's journey through the city on his bicycle. The video likely includes commentary or narration about the man's experience or the city's cycling infrastructure.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.368537902832031)
    The video is a collage of three images featuring two women posing together. The first image shows the women standing side by side, smiling and looking directly at the camera. The second image captures a close-up of the women, focusing on their facial expressions and the details of their attire. The third image is a wide shot of the women, showcasing their full outfits and the background setting. The style of the video is a simple, straightforward montage of the women, with no additional action or movement. The focus is on the women and their interaction with each other and the camera.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.373477935791016)
    The video captures a dynamic scene at a race track. The track is a long, curving road with a green and red striped pattern on the sides. The road is surrounded by a fence and a hill, providing a sense of depth and perspective. The sky above is cloudy, casting a soft light over the scene. In the distance, a car can be seen speeding along the track, adding a sense of motion and excitement to the image. The overall style of the video is dynamic and energetic, capturing the thrill and speed of the race track.
    zoom_out
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.400115966796875)
    The video captures a young man in a black shirt standing in a bustling convention center. He is looking directly at the camera with a surprised expression on his face. The convention center is filled with people walking around, some carrying backpacks and handbags. The lighting is bright and artificial, typical of indoor event spaces. The style of the video is candid and unposed, capturing a spontaneous moment in the midst of a busy event.
    pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.40855884552002)
    The video shows a man in a yellow shirt and a black helmet, wearing sunglasses and holding a cell phone to his ear. He is standing in a public area with other people in the background. The man appears to be engaged in a conversation on his phone. The style of the video is casual and candid, capturing a moment in the man's day. The focus is on the man and his interaction with his phone, with the background serving as a setting for the scene.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.417043685913086)
    The video features a man with long dreadlocks, wearing a green t-shirt and a black baseball cap, sitting in front of a microphone. He appears to be in a radio studio, with a red logo visible on the microphone. The man is speaking into the microphone, suggesting that he is either hosting a radio show or being interviewed. The style of the video is casual and informal, with a focus on the man and his interaction with the microphone. The setting is simple and uncluttered, allowing the viewer to focus on the man and his actions.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.43941879272461)
    In the video, a man with gray hair is seen sitting in the back seat of a car. He is wearing a black shirt and appears to be engaged in a conversation with another man who is sitting in the front seat. The man in the front seat is wearing a blue shirt and a hat. The car is moving, as indicated by the blurred background. The man in the back seat seems to be looking at the man in the front seat, suggesting an ongoing conversation between them. The overall style of the video is realistic, capturing a candid moment between two people in a car.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.439460754394531)
    The video features a man with a bald head and a beard, wearing glasses and a suit. He is seated in a studio setting with a blue background. The man appears to be engaged in a conversation or interview, as suggested by the presence of a microphone and a camera. The video also includes a Twitter hashtag, #StayTunedSTL, which indicates that the content may be related to a news or media outlet based in St. Louis. The style of the video is professional and polished, typical of a news or interview segment.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.454158782958984)
    The video features a man with a beard, wearing a black shirt with the word "WADE" on it, sitting in a chair and gesturing with his hands. He appears to be engaged in a conversation or interview. The background shows a barber shop setting with a barber chair and various barber tools. The style of the video suggests it could be a news segment or a promotional video for a barber shop.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.472391128540039)
    The video features a woman with tattoos on her arms, singing passionately into a microphone. She is dressed in a black top and her hair is styled in a short, dark bob. The background is a dark, blue-lit stage with arches, creating a dramatic and intimate atmosphere. The woman's performance is the focal point of the video, with her expressive facial expressions and the intensity of her singing conveying emotion and passion. The lighting and stage design enhance the overall mood and ambiance of the performance.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.47368335723877)
    The video shows a man holding a microphone and recording a conversation with another man on a city street. The man being interviewed is wearing a black jacket and a white hoodie, while the interviewer is dressed in a green jacket. The interviewer is holding a smartphone in his other hand. The background of the video shows a busy city street with other people walking by. The video has a timestamp of 0:41.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.489474296569824)
    The video captures a moment from a football game between the Broncos and the Bears. The main focus is a player from the Broncos team, wearing a white jersey with the number 17 and an orange cap. He is standing on the sidelines, arms crossed, watching the game intently. The background is filled with other players, some of whom are also wearing orange caps, indicating they are part of the Broncos team. The image is slightly blurred, suggesting movement and action on the field. The overall style of the video is dynamic and captures the intensity of the game.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.56825065612793)
    In the video, a young woman is seen posing for the camera with a radiant smile. She is elegantly dressed in a pink sequined jacket, which adds a touch of glamour to her appearance. Her hair is styled in loose waves, complementing her overall look. She is wearing pink earrings that match her outfit, and her makeup is done in a subtle yet sophisticated manner. The background features a red and white backdrop, which contrasts beautifully with her pink ensemble. The video captures her in three different frames, each showcasing her beauty and style.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.590845108032227)
    The video is a news segment featuring a man and a woman in a studio setting. The man is wearing a suit and tie, while the woman is dressed in a floral print top. They are seated behind a desk with a cityscape in the background. The man is holding a microphone, suggesting that he is the one speaking. The woman is also holding a microphone, indicating that she is also part of the conversation. The overall style of the video is professional and polished, typical of a news broadcast.
    pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.59793472290039)
    The video shows a person's hand interacting with a car's touch screen display. The hand is seen pressing a button on the screen, which is located in the center console of the car. The display shows various icons and information, including the time and temperature. The car's interior is visible in the background, with the dashboard and air vents visible. The style of the video is a close-up shot, focusing on the hand and the touch screen display. The video captures the action of the hand pressing the button, and the response of the display. The video does not show any other objects or actions.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.607828140258789)
    In the video, a man in a blue shirt stands confidently in a garage, gesturing with his hands as he speaks. Behind him, a blue car is parked on a lift, its hood open, revealing the engine. The garage is well-lit, with various tools and equipment scattered around, indicating a space for vehicle maintenance and repair. The man's attire and the setting suggest that he might be a mechanic or a car enthusiast. The video captures a moment of explanation or demonstration, possibly related to the car's engine or maintenance.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.621119499206543)
    The video shows a man working on a car engine in a garage. The man is wearing a gray t-shirt and is focused on the task at hand. He is using a tool to work on the engine, which is silver and black in color. The garage is filled with various tools and equipment, indicating that it is a well-equipped workspace. The man's actions suggest that he is experienced in car maintenance and is likely performing a repair or upgrade on the engine. The overall style of the video is a real-life, documentary-style footage, capturing the man's work in a garage setting.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=1.0403149127960205, certainty=None, score=0.0, explain_score='', is_consistent=None, rerank_score=-10.663298606872559)
    

> 观察到结果确实如我们想要的意义，一个带有房间窗户户外食物元素的内容召回
> 
> 不过对于这种召回仍然有些瑕疵，具体还是因为我们仍然无法度量，我们的任务在这几个元素的占比
> 
> 当空间标量无法度量（有聚集）时，这个召回策略仍然不可用
>
> 重排序仍然有个问题，当我们的数据样本很少，又需要把样本按照标量标记空间距离，这时一个准确的标量空间应该是尽可能的在一个聚集内

### 方式一

目标过滤



```python
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter

## 查找关于食物特写的描述
response = jeopardy.query.bm25(
    query="Food",  
    limit=100,
    filters=Filter.by_property("caption").equal("a close-up view"),
    query_properties=["Caption"],
    return_metadata=MetadataQuery(score=True, explain_score=True),
)


## 查找关于食物牛油果的描述
response = jeopardy.query.bm25(
    query="Food",  
    limit=100,
    filters=Filter.by_property("caption").equal("avocado slices"),
    query_properties=["Caption"],
    return_metadata=MetadataQuery(score=True, explain_score=True),
)


for o in response.objects:
    print(o.uuid)
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata)
```

    a5528dee-c3c1-452a-a554-737e52f05ff6
    The video shows a close-up of a plate of food, which includes a serving of rice, beans, and avocado slices. The food is presented on a green plate with a striped pattern. The style of the video is simple and straightforward, focusing on the food without any additional context or background. The camera angle is slightly elevated, providing a clear view of the food on the plate. The lighting is bright, highlighting the colors and textures of the food. The video does not contain any text or additional elements, and the focus is solely on the plate of food.
    Undetermined
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=2.2844784259796143, explain_score=', BM25F_food_frequency:6, BM25F_food_propLength:53', is_consistent=None, rerank_score=None)
    1466a12b-be98-410c-a9e7-3118c1b6dfdd
    The video shows a close-up of a plate of food on a wooden table. The food consists of two slices of bread topped with avocado, sliced strawberries, and crumbled feta cheese. The strawberries are bright red and fresh, and the avocado is creamy and green. The feta cheese adds a touch of white to the colorful dish. The plate is white, which contrasts nicely with the vibrant colors of the food. The wooden table provides a warm and rustic backdrop to the meal. The video is likely a food blog or a cooking tutorial, showcasing a healthy and delicious meal.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=2.096710205078125, explain_score=', BM25F_food_frequency:4, BM25F_food_propLength:56', is_consistent=None, rerank_score=None)
    ac4a500f-622f-474c-bd14-5a1d5a164d60
    The video is a close-up, high-resolution, and colorful food presentation. It features a white plate with slices of toasted bread topped with sliced avocado, boiled eggs, and chopped nuts. The eggs are halved and arranged in a circular pattern, with the yolks facing up. The avocado slices are arranged in a fan-like pattern, and the nuts are sprinkled on top of the eggs and avocado. The food is garnished with black pepper. The style of the video is gourmet and appetizing, with a focus on the textures and colors of the ingredients. The lighting is bright and even, highlighting the freshness and quality of the food. The video is likely intended for a food blog or a cooking channel, showcasing a simple yet elegant breakfast or brunch dish.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=2.027494430541992, explain_score=', BM25F_food_frequency:4, BM25F_food_propLength:71', is_consistent=None, rerank_score=None)
    03283236-daec-4156-9910-562ea37b59ea
    The video is a close-up, high-resolution, and colorful food presentation. It features a white plate with slices of toasted bread topped with sliced avocado, boiled eggs, and chopped nuts. The eggs are halved and arranged in a circular pattern, with the yolks facing up. The avocado slices are arranged in a fan-like pattern, and the nuts are sprinkled on top of the eggs and avocado. The food is garnished with black pepper. The style of the video is gourmet and appetizing, with a focus on the textures and colors of the ingredients. The lighting is bright and even, highlighting the freshness and quality of the food. The video is likely intended for a food blog or a cooking channel, showcasing a simple yet elegant breakfast or brunch dish.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=2.027494430541992, explain_score=', BM25F_food_frequency:4, BM25F_food_propLength:71', is_consistent=None, rerank_score=None)
    0b183c3d-268c-4afb-a92d-3a5c5f296689
    The video captures a close-up view of a plate of food, focusing on the vibrant colors and textures of the ingredients. The plate is filled with a variety of ingredients, including chunks of yellow pineapple, slices of green avocado, and pieces of brown meat. The pineapple is bright yellow and appears to be fresh, while the avocado is a deep green color, indicating it is ripe. The meat is brown and appears to be cooked, adding a savory element to the dish. The ingredients are arranged in a visually appealing manner, with the pineapple and avocado taking center stage. The video is shot in a realistic style, with a focus on the food and its presentation. The colors and textures of the ingredients are vividly captured, making the dish look appetizing and inviting. The video does not contain any text or narration, allowing the viewer to focus solely on the food. The overall style of the video is simple and straightforward, with a focus on the food and its presentation.
    zoom_out
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.9838346242904663, explain_score=', BM25F_food_frequency:4, BM25F_food_propLength:81', is_consistent=None, rerank_score=None)
    46b42725-fcc9-44c9-8384-c8f683d78a2d
    In the video, two men are enjoying a meal together outdoors. They are standing in front of a food truck, which is decorated with a colorful mural of a beach scene. The men are holding white plates filled with food, and they are eating slices of pizza topped with avocado. The pizza looks delicious and fresh. The men are dressed casually, and they seem to be having a good time. The food truck is parked on a street, and the atmosphere is relaxed and friendly. The video captures a moment of enjoyment and camaraderie between the two men.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.940184235572815, explain_score=', BM25F_food_frequency:3, BM25F_food_propLength:57', is_consistent=None, rerank_score=None)
    938c7e52-5a48-494f-a901-4eecbdc3cd53
    The video shows a close-up of a wooden cutting board with three slices of bread topped with various ingredients. The first slice has a layer of creamy white cheese, sliced cherry tomatoes, and chopped green herbs. The second slice features a mix of chopped red and green bell peppers, crumbled white cheese, and a sprinkle of green herbs. The third slice is topped with sliced avocado, chopped red onions, and a drizzle of olive oil. The cutting board is placed on a wooden surface, and the background is blurred, focusing the viewer's attention on the food. The style of the video is a simple, straightforward food presentation, likely intended for a recipe or food blog. The lighting is bright and even, highlighting the textures and colors of the ingredients.
    pan_right+tilt_down
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.8617665767669678, explain_score=', BM25F_food_frequency:3, BM25F_food_propLength:72', is_consistent=None, rerank_score=None)
    5869e03d-33b3-40ea-b3b6-b69ffad87597
    The video is a close-up of a meal being eaten with a white plastic fork. The meal consists of a variety of ingredients, including shredded meat, lime slices, avocado, and chopped onions. The fork is used to pick up the food, and the person eating the meal is not visible in the video. The style of the video is simple and straightforward, focusing on the food and the fork without any additional context or background. The video does not contain any text or narration. The overall impression is that of a casual, everyday meal being enjoyed.
    tilt_up+zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.700600028038025, explain_score=', BM25F_food_propLength:56, BM25F_food_frequency:2', is_consistent=None, rerank_score=None)
    3d551466-e450-4f7f-aea6-5d686960d006
    In the video, a man and a woman are seated at a dining table, enjoying a meal together. The man is wearing a blue shirt, while the woman is dressed in a green dress. They are both seated on blue chairs. The table is set with plates of food, including sandwiches and avocado slices. The woman is seen serving the food to the man, who is seated across from her. The scene is set in a kitchen, with a counter and a sink visible in the background. The overall atmosphere of the video is casual and relaxed, capturing a moment of shared enjoyment between the two individuals.
    zoom_in
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.675851821899414, explain_score=', BM25F_food_frequency:2, BM25F_food_propLength:60', is_consistent=None, rerank_score=None)
    a45d81b2-d75a-4f78-a544-897d1f28c50f
    The video shows a person preparing a sandwich on a wooden cutting board. The person is seen placing a slice of bread on the board, followed by a slice of avocado, a tomato, and a leaf of spinach. The sandwich is then assembled with the ingredients, and the person is seen holding a knife, presumably to cut the sandwich. The cutting board is placed on a white countertop, and there are bowls containing additional ingredients, such as more avocado slices and tomatoes. The style of the video is a simple, straightforward food preparation video, with a focus on the sandwich-making process. The lighting is bright and even, highlighting the colors of the ingredients and the texture of the wooden cutting board. The person's hands are visible, but their face is not shown, keeping the focus on the food preparation.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.6227184534072876, explain_score=', BM25F_food_frequency:2, BM25F_food_propLength:69', is_consistent=None, rerank_score=None)
    c65467e8-144c-41aa-a121-b07869ca9c4a
    In the video, a person is preparing a sandwich on a wooden table. The sandwich is made with slices of bread, avocado, and bacon. The person is using a spoon to scoop corn from a bowl and add it to the sandwich. There are also bowls of rice and corn on the table. The person is wearing a blue and white striped shirt. The video captures the process of making the sandwich, from the preparation of the ingredients to the final assembly. The style of the video is casual and homey, with a focus on the food preparation process.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.2752552032470703, explain_score=', BM25F_food_frequency:1, BM25F_food_propLength:50', is_consistent=None, rerank_score=None)
    aa7f45b6-a770-413e-a000-0df29d81e6d7
    In the video, a person is preparing a sandwich on a wooden table. The sandwich is made with slices of bread, avocado, and bacon. The person is using a spoon to scoop corn from a bowl and add it to the sandwich. There are also bowls of rice and corn on the table. The person is wearing a blue and white striped shirt. The video captures the process of making the sandwich, from the preparation of the ingredients to the final assembly. The style of the video is casual and homey, with a focus on the food preparation process.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.2752552032470703, explain_score=', BM25F_food_frequency:1, BM25F_food_propLength:50', is_consistent=None, rerank_score=None)
    76efeea6-ba6b-4d31-8c5f-06b7c2698621
    In the video, a person is seen preparing a dish on a wooden table. The dish consists of two slices of bread topped with a variety of ingredients, including tomatoes, lettuce, and avocado. The person is using a squeeze bottle to drizzle a sauce over the dish. The sauce appears to be a dark color, possibly a vinaigrette or a similar type of dressing. The person is wearing a striped shirt, and the overall style of the video suggests a casual, home-cooking atmosphere. The focus is on the food preparation, with the person's actions and the ingredients being the main subjects of the video.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.2020426988601685, explain_score=', BM25F_food_frequency:1, BM25F_food_propLength:61', is_consistent=None, rerank_score=None)
    ecad7dd0-ff29-4fae-8387-5975acd2cac6
    The video shows a creative and colorful food platter being prepared. The platter is designed to resemble a cake, with layers of various ingredients. The first layer consists of sliced hard-boiled eggs, arranged in a circular pattern. The second layer features slices of orange, arranged in a similar circular pattern. The third layer consists of slices of ham, arranged in a circular pattern. The fourth layer features slices of cucumber, arranged in a circular pattern. The fifth layer consists of slices of tomato, arranged in a circular pattern. The sixth layer features slices of cheese, arranged in a circular pattern. The seventh layer consists of slices of onion, arranged in a circular pattern. The eighth layer consists of slices of bell pepper, arranged in a circular pattern. The ninth layer consists of slices of avocado, arranged in a circular pattern. The tenth layer consists of slices of lemon, arranged in a circular pattern. The eleventh layer consists of slices of lime, arranged in a circular pattern. The twelfth layer consists of slices of orange, arranged in a circular pattern. The thirteenth layer consists of slices of tomato, arranged in a circular pattern. The fourteenth layer consists of slices of cucumber, arranged in a circular pattern. The fifteenth layer consists of slices of onion, arranged in a circular pattern. The sixteenth layer consists of slices of bell pepper, arranged in a circular pattern. The seventeenth layer consists of slices of avocado, arranged in a circular pattern. The
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.1958016157150269, explain_score=', BM25F_food_frequency:1, BM25F_food_propLength:62', is_consistent=None, rerank_score=None)
    3c0c80ed-dd27-439b-95b6-ee786245278b
    The video shows a person preparing a sandwich with various ingredients. The sandwich is made with a slice of bread, topped with a layer of creamy spread, possibly avocado or cream cheese. The spread is garnished with a sprig of fresh green leaves, possibly parsley or cilantro, and a few slices of ripe red tomato. The sandwich is held in the person's hand, which is visible in the foreground of the image. In the background, there are additional ingredients, including a whole avocado, a bowl of green spread, and a few cherry tomatoes. The overall style of the video is simple and clean, focusing on the preparation of the sandwich and the freshness of the ingredients. The lighting is bright and even, highlighting the colors and textures of the food. The video does not contain any text or additional elements, keeping the focus solely on the sandwich and the preparation process.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.098815679550171, explain_score=', BM25F_food_frequency:1, BM25F_food_propLength:79', is_consistent=None, rerank_score=None)
    04b5b065-1e23-45f5-b172-592a512ed347
    The video shows a person preparing a sandwich with various ingredients. The sandwich is made with a slice of bread, topped with a layer of creamy spread, possibly avocado or cream cheese. The spread is garnished with a sprig of fresh green leaves, possibly parsley or cilantro, and a few slices of ripe red tomato. The sandwich is held in the person's hand, which is visible in the foreground of the image. In the background, there are additional ingredients, including a whole avocado, a bowl of green spread, and a few cherry tomatoes. The overall style of the video is simple and clean, focusing on the preparation of the sandwich and the freshness of the ingredients. The lighting is bright and even, highlighting the colors and textures of the food. The video does not contain any text or additional elements, keeping the focus solely on the sandwich and the preparation process.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.098815679550171, explain_score=', BM25F_food_frequency:1, BM25F_food_propLength:79', is_consistent=None, rerank_score=None)
    

### 方式二

目标operators偏移，修改搜索词在空间向量的坐标，可控制的操作方法如，在空间距离上修改词的距离，远离或者接近

https://weaviate.io/developers/weaviate/api/graphql/search-operators#example-ii

#### Semantic Path 语义路径 
仅适用于 text2vec-contextionary 模块

注意：仅当将 nearText: {} 操作符设置为探索术语时，才能构建语义路径，因为探索术语代表路径的开始，每个搜索结果代表路径的结束。由于 nearText: {} 查询目前仅在 GraphQL 中可行，因此 semanticPath 在 REST API 中不可用。


```python
# Semantic path is not yet supported by the V4 client. Please use a raw GraphQL query instead.
response = client.graphql_raw_query(
  """
  {
    Get {
      OpenVidContext(
        nearText:{
          concepts: ["food"], 
          distance: 0.23, 
          moveAwayFrom: {
            concepts: ["finance"],
            force: 0.45
          },
          moveTo: {
            concepts: ["apples", "food"],
            force: 0.85
          }
        }
      ) {
        caption
        _additional {
          semanticPath {
            path {
              concept
              distanceToNext
              distanceToPrevious
              distanceToQuery
              distanceToResult
            }
          }
        }
      }
    }
  }
  """
)
response
```




    _RawGQLReturn(aggregate={}, explore={}, get={'OpenVidContext': [{'_additional': {'semanticPath': {'path': [{'concept': 'food', 'distanceToNext': 0.20903677, 'distanceToPrevious': None, 'distanceToQuery': 0.043871343, 'distanceToResult': 0.23713183}, {'concept': 'foods', 'distanceToNext': 0.39609814, 'distanceToPrevious': 0.20903677, 'distanceToQuery': 0.20665145, 'distanceToResult': 0.3520723}, {'concept': 'milk', 'distanceToNext': 0.4368891, 'distanceToPrevious': 0.39609814, 'distanceToQuery': 0.369789, 'distanceToResult': 0.35984027}, {'concept': 'tastes', 'distanceToNext': 0.35711163, 'distanceToPrevious': 0.4368891, 'distanceToQuery': 0.38843882, 'distanceToResult': 0.32041246}, {'concept': 'fresh', 'distanceToNext': None, 'distanceToPrevious': 0.35711163, 'distanceToQuery': 0.33576888, 'distanceToResult': 0.25575793}]}}, 'caption': 'The video shows a close-up of a plate of food being eaten. The plate contains a fried egg, a slice of ham, and baked beans. The person eating the food is using a fork and knife. The food is being eaten in a casual setting, and the person is enjoying their meal. The video captures the details of the food and the eating process, providing a sense of the meal being enjoyed.'}, {'_additional': {'semanticPath': {'path': [{'concept': 'food', 'distanceToNext': 0.35004628, 'distanceToPrevious': None, 'distanceToQuery': 0.043871343, 'distanceToResult': 0.26622736}, {'concept': 'delicious', 'distanceToNext': 0.11825091, 'distanceToPrevious': 0.35004628, 'distanceToQuery': 0.28628564, 'distanceToResult': 0.25686538}, {'concept': 'tasty', 'distanceToNext': 0.5612864, 'distanceToPrevious': 0.11825091, 'distanceToQuery': 0.32107842, 'distanceToResult': 0.31186754}, {'concept': 'grocery', 'distanceToNext': 0.5547122, 'distanceToPrevious': 0.5612864, 'distanceToQuery': 0.3710547, 'distanceToResult': 0.39492106}, {'concept': 'ingredients', 'distanceToNext': 0.38889855, 'distanceToPrevious': 0.5547122, 'distanceToQuery': 0.3366087, 'distanceToResult': 0.30253226}, {'concept': 'fresh', 'distanceToNext': None, 'distanceToPrevious': 0.38889855, 'distanceToQuery': 0.33576888, 'distanceToResult': 0.19813716}]}}, 'caption': "The video is a vibrant and colorful display of a refrigerator filled with a variety of food items. The refrigerator is well-stocked with fresh fruits, vegetables, and beverages. The fruits include oranges, bananas, and grapes, while the vegetables include tomatoes and broccoli. The beverages include bottles of juice and water. The refrigerator also contains a variety of desserts, including cupcakes and a jar of berries. The overall style of the video is bright and cheerful, with a focus on the freshness and variety of the food items. The video is likely intended to showcase the abundance and diversity of the refrigerator's contents."}, {'_additional': {'semanticPath': {'path': [{'concept': 'food', 'distanceToNext': 0.24659812, 'distanceToPrevious': None, 'distanceToQuery': 0.043871343, 'distanceToResult': 0.23637164}, {'concept': 'eat', 'distanceToNext': 0.40395093, 'distanceToPrevious': 0.24659812, 'distanceToQuery': 0.22321028, 'distanceToResult': 0.25413448}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.40395093, 'distanceToQuery': 0.3645923, 'distanceToResult': 0.37023735}]}}, 'caption': 'The video captures a close-up view of a plate of food being eaten. The plate contains a colorful dish with various ingredients, including meat, vegetables, and herbs. A spoon is used to scoop up the food, and the person eating the dish is using a knife to cut the meat. The food is being eaten in a restaurant setting, with a tablecloth visible in the background. The style of the video is realistic and it captures the details of the food and the eating process.'}, {'_additional': {'semanticPath': {'path': [{'concept': 'food', 'distanceToNext': 0.39643884, 'distanceToPrevious': None, 'distanceToQuery': 0.043871343, 'distanceToResult': 0.25219238}, {'concept': 'treats', 'distanceToNext': 0.444021, 'distanceToPrevious': 0.39643884, 'distanceToQuery': 0.3645923, 'distanceToResult': 0.3569348}, {'concept': 'tastes', 'distanceToNext': None, 'distanceToPrevious': 0.444021, 'distanceToQuery': 0.38843882, 'distanceToResult': 0.29319608}]}}, 'caption': "The video shows a person eating a bowl of soup. The soup contains meat, vegetables, and noodles. The person is using a spoon to eat the soup. The soup is served in a bowl that is placed on a table. The person is wearing a red shirt. The soup appears to be hot and the person is enjoying it. The spoon is made of metal and is being used to scoop up the soup. The vegetables in the soup include carrots and green beans. The meat in the soup is likely chicken or beef. The noodles in the soup are likely egg noodles. The table is made of wood and is brown in color. The person is sitting at the table while eating the soup. The soup is being eaten in a casual setting. The person is likely enjoying a meal at home or in a restaurant. The soup is likely a main course and is being eaten during lunch or dinner. The person is likely using the spoon to eat the soup because it is a convenient utensil for eating soup. The person is likely enjoying the soup because it is a warm and comforting meal. The soup is likely a popular dish in the person's culture or region. The person is likely eating the soup because it is a delicious and satisfying meal. The soup is likely a traditional dish that has been passed down through generations. The person is likely eating the soup because it is a nutritious and healthy meal. The soup is likely a popular dish in the person's community or social circle. The"}, {'_additional': {'semanticPath': {'path': [{'concept': 'food', 'distanceToNext': 0.35769427, 'distanceToPrevious': None, 'distanceToQuery': 0.043871343, 'distanceToResult': 0.2713257}, {'concept': 'snack', 'distanceToNext': 0.2759233, 'distanceToPrevious': 0.35769427, 'distanceToQuery': 0.31253606, 'distanceToResult': 0.32785183}, {'concept': 'meals', 'distanceToNext': 0.09339923, 'distanceToPrevious': 0.2759233, 'distanceToQuery': 0.24887824, 'distanceToResult': 0.25634867}, {'concept': 'meal', 'distanceToNext': 0.40922564, 'distanceToPrevious': 0.09339923, 'distanceToQuery': 0.23755962, 'distanceToResult': 0.22967243}, {'concept': 'treats', 'distanceToNext': 0.41248167, 'distanceToPrevious': 0.40922564, 'distanceToQuery': 0.3645923, 'distanceToResult': 0.35816556}, {'concept': 'taste', 'distanceToNext': 0.10029024, 'distanceToPrevious': 0.41248167, 'distanceToQuery': 0.3283019, 'distanceToResult': 0.26296395}, {'concept': 'tastes', 'distanceToNext': 0.21134079, 'distanceToPrevious': 0.10029024, 'distanceToQuery': 0.38843882, 'distanceToResult': 0.2996446}, {'concept': 'flavor', 'distanceToNext': 0.30955607, 'distanceToPrevious': 0.21134079, 'distanceToQuery': 0.40333366, 'distanceToResult': 0.3270157}, {'concept': 'spices', 'distanceToNext': None, 'distanceToPrevious': 0.30955607, 'distanceToQuery': 0.38909608, 'distanceToResult': 0.36663926}]}}, 'caption': 'The video is a compilation of three different meals, each presented in a metal container. The meals are diverse and colorful, featuring a variety of fruits, vegetables, and salads. The first meal includes a mix of fruits such as bananas, strawberries, and blueberries, as well as a salad with lettuce and carrots. The second meal consists of a salad with tomatoes and cucumbers, along with a side of crackers. The third meal features a salad with lettuce, tomatoes, and carrots, accompanied by a side of fruit. The style of the video is simple and straightforward, focusing on the presentation of the meals without any additional context or narrative. The meals are arranged neatly in the containers, and the colors of the fruits and vegetables are vibrant and appealing. The overall impression is one of healthy and nutritious eating.'}, {'_additional': {'semanticPath': {'path': [{'concept': 'food', 'distanceToNext': 0.24659812, 'distanceToPrevious': None, 'distanceToQuery': 0.043871343, 'distanceToResult': 0.24235499}, {'concept': 'eat', 'distanceToNext': 0.20734924, 'distanceToPrevious': 0.24659812, 'distanceToQuery': 0.22321028, 'distanceToResult': 0.25023776}, {'concept': 'meal', 'distanceToNext': 0.40423477, 'distanceToPrevious': 0.20734924, 'distanceToQuery': 0.23755962, 'distanceToResult': 0.2235896}, {'concept': 'tastes', 'distanceToNext': 0.10029024, 'distanceToPrevious': 0.40423477, 'distanceToQuery': 0.38843882, 'distanceToResult': 0.29688174}, {'concept': 'taste', 'distanceToNext': 0.30589926, 'distanceToPrevious': 0.10029024, 'distanceToQuery': 0.3283019, 'distanceToResult': 0.24096692}, {'concept': 'fresh', 'distanceToNext': None, 'distanceToPrevious': 0.30589926, 'distanceToQuery': 0.33576888, 'distanceToResult': 0.25330514}]}}, 'caption': 'The video captures a person enjoying a meal at a street food stall. The person is using chopsticks to eat from a bowl of soup, which contains noodles, vegetables, and meat. The soup is served in a yellow bowl, and there is a small bowl of seasoning nearby. The table is covered with a white tablecloth, and there are other dishes and condiments in the background. The setting suggests a casual and relaxed dining experience, with the focus on the delicious food being enjoyed.'}]}, errors=None)




```python
 # Semantic path is not yet supported by the V4 client. Please use a raw GraphQL query instead.
response = client.graphql_raw_query(
  """
{
  Get {
    OpenVidContext(
      nearText: {
        concepts: ["food", "strawberry"], 
        moveAwayFrom: {
          concepts: ["finance"],
          force: 0.45
        },
        moveTo: {
          concepts: ["strawberry", "food", "ice cream"],
          force: 0.85
        }
      }, 
      limit: 25
    ) {
      video
      caption
      _additional {
        semanticPath {
          path {
            concept
            distanceToNext
            distanceToPrevious
            distanceToQuery
            distanceToResult
          }
        }
      }
    }
  }
}


  """
)
response
```




    _RawGQLReturn(aggregate={}, explore={}, get={'OpenVidContext': [{'_additional': {'semanticPath': {'path': [{'concept': 'tastes', 'distanceToNext': 0.10029024, 'distanceToPrevious': None, 'distanceToQuery': 0.36325133, 'distanceToResult': 0.24331182}, {'concept': 'taste', 'distanceToNext': 0.2607085, 'distanceToPrevious': 0.10029024, 'distanceToQuery': 0.2937745, 'distanceToResult': 0.18520582}, {'concept': 'sweet', 'distanceToNext': 0.39880985, 'distanceToPrevious': 0.2607085, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.21114302}, {'concept': 'cheese', 'distanceToNext': 0.21689123, 'distanceToPrevious': 0.39880985, 'distanceToQuery': 0.2789914, 'distanceToResult': 0.2870338}, {'concept': 'bread', 'distanceToNext': None, 'distanceToPrevious': 0.21689123, 'distanceToQuery': 0.33978528, 'distanceToResult': 0.3246225}]}}, 'caption': 'The video captures a close-up view of a delicious ice cream sundae being enjoyed by a person. The sundae is a delightful mix of flavors and textures, with a generous scoop of vanilla ice cream at the bottom, topped with a layer of whipped cream. Adding to the visual appeal and taste, the whipped cream is drizzled with a vibrant red strawberry sauce.   The sundae is further adorned with fresh fruit, including juicy orange slices and ripe strawberries, which add a pop of color and a refreshing tang to the sweet treat. The person enjoying the sundae is holding a spoon, ready to take a bite, indicating the anticipation and enjoyment of the dessert. The overall style of the video is appetizing and inviting, making the viewer crave a taste of the ice cream sundae.', 'video': 'BZhj-4k2hHI_28_0to238.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.27576005, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.27547157}, {'concept': 'delicious', 'distanceToNext': 0.2523045, 'distanceToPrevious': 0.27576005, 'distanceToQuery': 0.25429422, 'distanceToResult': 0.24283177}, {'concept': 'baked', 'distanceToNext': 0.41837227, 'distanceToPrevious': 0.2523045, 'distanceToQuery': 0.3318305, 'distanceToResult': 0.29578602}, {'concept': 'flavour', 'distanceToNext': 0.33818376, 'distanceToPrevious': 0.41837227, 'distanceToQuery': 0.39830786, 'distanceToResult': 0.41310614}, {'concept': 'fresh', 'distanceToNext': None, 'distanceToPrevious': 0.33818376, 'distanceToQuery': 0.3249395, 'distanceToResult': 0.23827165}]}}, 'caption': 'The video shows a person eating a bowl of fruit and yogurt. The bowl is filled with strawberries, bananas, and granola. The person is using an orange spoon to scoop the fruit and yogurt. The bowl is placed on a white surface, and there is a laptop in the background. The person is enjoying their meal, and the video captures the delicious and healthy nature of the food.', 'video': '6sqG1ObKfP4_29_0to221.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'freshly', 'distanceToNext': 0.21529531, 'distanceToPrevious': None, 'distanceToQuery': 0.42613184, 'distanceToResult': 0.35219765}, {'concept': 'fresh', 'distanceToNext': 0.48085022, 'distanceToPrevious': 0.21529531, 'distanceToQuery': 0.3249395, 'distanceToResult': 0.2707259}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.48085022, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.35048842}]}}, 'caption': 'The video shows a close-up of a table with three chocolate desserts, each with a unique design. The first dessert is a chocolate cup filled with whipped cream and topped with a chocolate-covered strawberry. The second dessert is a chocolate cup filled with whipped cream and topped with a chocolate-covered strawberry. The third dessert is a chocolate cup filled with whipped cream and topped with a chocolate-covered strawberry. The desserts are placed on a wooden table, and the camera angle is from above, providing a clear view of the desserts. The style of the video is a simple, straightforward presentation of the desserts, with no additional elements or distractions.', 'video': 'Dcdh6ThJ4hA_0_0to136.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'chocolate', 'distanceToNext': 0.2884171, 'distanceToPrevious': None, 'distanceToQuery': 0.28091043, 'distanceToResult': 0.23370486}, {'concept': 'dessert', 'distanceToNext': 0.41062486, 'distanceToPrevious': 0.2884171, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.32281667}, {'concept': 'sweet', 'distanceToNext': None, 'distanceToPrevious': 0.41062486, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.25972527}]}}, 'caption': 'The video captures the delightful process of enjoying a chocolate ice cream sundae. The sundae is presented in a tall glass, filled with layers of rich chocolate ice cream, creamy vanilla ice cream, and a generous topping of crushed chocolate cookies. A spoon is inserted into the sundae, ready to scoop up the delicious treat. The sundae is placed on a brown plate, which contrasts nicely with the vibrant colors of the ice cream and cookies. The entire scene is set against a blurred background, drawing focus to the main subject of the video - the mouth-watering chocolate ice cream sundae.', 'video': '3njnhFP-ul4_99_0to123.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'chocolate', 'distanceToNext': 0.35633332, 'distanceToPrevious': None, 'distanceToQuery': 0.28091043, 'distanceToResult': 0.29102677}, {'concept': 'pie', 'distanceToNext': 0.5094519, 'distanceToPrevious': 0.35633332, 'distanceToQuery': 0.38405937, 'distanceToResult': 0.3245089}, {'concept': 'treats', 'distanceToNext': 0.4360329, 'distanceToPrevious': 0.5094519, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.36402297}, {'concept': 'sweet', 'distanceToNext': None, 'distanceToPrevious': 0.4360329, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.26059252}]}}, 'caption': "The video is a close-up of a dessert being eaten. The dessert is a large scoop of ice cream topped with whipped cream and a generous amount of fruit, possibly peaches or pineapple. The fruit is caramelized, giving it a golden-brown color. The ice cream is placed in a bowl, and a spoon is visible, indicating that the dessert is being enjoyed. The style of the video is casual and appetizing, with a focus on the dessert's texture and color. The background is blurred, drawing attention to the dessert. The video is likely meant to showcase the dessert's deliciousness and appeal to viewers' taste buds.", 'video': 'H6MW6SGP5tg_27_0to109.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': None, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.28390723}]}}, 'caption': "The video shows a person preparing a dessert, specifically a peach with a scoop of ice cream and a drizzle of caramel sauce. The peach is halved and filled with ice cream, and the caramel sauce is being poured over the top. The person is using a spoon to scoop the ice cream and drizzle the sauce. The dessert is placed on a black tray, and there are crumbs scattered around the tray. The style of the video is a close-up shot, focusing on the dessert and the person's hands as they prepare it. The lighting is bright, highlighting the colors of the peach, ice cream, and caramel sauce. The video captures the process of preparing the dessert, from scooping the ice cream to drizzling the sauce, and the final product is presented as a delicious and visually appealing treat.", 'video': 'HneZ2wDm9rQ_16_51to201.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'delicious', 'distanceToNext': 0.25202137, 'distanceToPrevious': None, 'distanceToQuery': 0.25429422, 'distanceToResult': 0.21219867}, {'concept': 'cooked', 'distanceToNext': None, 'distanceToPrevious': 0.25202137, 'distanceToQuery': 0.37585306, 'distanceToResult': 0.29901868}]}}, 'caption': 'The video shows a delicious dessert being prepared and served. The dessert consists of a waffle, strawberries, whipped cream, and a drizzle of syrup. The waffle is golden brown and crispy, while the strawberries are fresh and red. The whipped cream is white and fluffy, and the syrup is a rich, dark brown color. The dessert is presented on a white plate, which contrasts nicely with the vibrant colors of the ingredients. The spoon is placed on the side of the plate, ready to be used. The overall style of the video is simple and elegant, focusing on the presentation of the dessert and the enjoyment of the viewer.', 'video': '1aMpykKRJf8_2_0to441.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'tastes', 'distanceToNext': 0.10029024, 'distanceToPrevious': None, 'distanceToQuery': 0.36325133, 'distanceToResult': 0.2880414}, {'concept': 'taste', 'distanceToNext': 0.2607085, 'distanceToPrevious': 0.10029024, 'distanceToQuery': 0.2937745, 'distanceToResult': 0.22625637}, {'concept': 'sweet', 'distanceToNext': 0.4360329, 'distanceToPrevious': 0.2607085, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.2359063}, {'concept': 'treats', 'distanceToNext': 0.37915152, 'distanceToPrevious': 0.4360329, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.3813895}, {'concept': 'snacks', 'distanceToNext': 0.08443117, 'distanceToPrevious': 0.37915152, 'distanceToQuery': 0.32523292, 'distanceToResult': 0.42018306}, {'concept': 'snack', 'distanceToNext': 0.37263328, 'distanceToPrevious': 0.08443117, 'distanceToQuery': 0.28378922, 'distanceToResult': 0.36083907}, {'concept': 'peanut', 'distanceToNext': None, 'distanceToPrevious': 0.37263328, 'distanceToQuery': 0.35662127, 'distanceToResult': 0.38077044}]}}, 'caption': 'In the video, a person is seen dipping strawberries into a cup of chocolate sauce. The person is holding the cup of chocolate sauce in their hand, and the strawberries are being dipped one by one. The chocolate sauce is rich and dark, and the strawberries are bright red and fresh. The person is using a spoon to dip the strawberries, and the spoon is visible in the cup of chocolate sauce. The person is wearing a white apron, and the background is a wooden table. The video captures the process of dipping the strawberries in the chocolate sauce, and it is a simple yet delightful scene.', 'video': 'FkZd5XPDI2w_10_546to817.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.41062486, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.23315936}, {'concept': 'dessert', 'distanceToNext': None, 'distanceToPrevious': 0.41062486, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.3170706}]}}, 'caption': 'The video captures a delightful dessert scene. A white bowl filled with a strawberry shortcake sits on a wooden table. The shortcake is topped with fresh strawberries and blackberries, adding a pop of color to the dish. A silver spoon is used to scoop up a bite of the dessert, revealing the layers of the shortcake and the juicy berries. The wooden table provides a rustic backdrop to the scene, while the strawberries and blackberries scattered around the bowl add a touch of freshness. The overall style of the video is simple yet elegant, focusing on the delicious dessert and the careful preparation involved in serving it.', 'video': '3Znh1WmcgzQ_0_0to125.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'pie', 'distanceToNext': 0.5480791, 'distanceToPrevious': None, 'distanceToQuery': 0.38405937, 'distanceToResult': 0.33599615}, {'concept': 'juice', 'distanceToNext': 0.56172013, 'distanceToPrevious': 0.5480791, 'distanceToQuery': 0.35646623, 'distanceToResult': 0.3792115}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.56172013, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.36711287}]}}, 'caption': 'The video is a delightful culinary journey featuring a stack of pancakes with strawberries and whipped cream. The pancakes are golden brown, topped with a dusting of powdered sugar, and garnished with fresh strawberries and a dollop of whipped cream. The strawberries are vibrant red, with their green leaves adding a touch of color contrast. The whipped cream is fluffy and white, adding a creamy texture to the dish. The plate holding the pancakes is blue, providing a cool contrast to the warm tones of the pancakes and strawberries. The background is a blurred pink, which could be a tablecloth or a wall, adding a soft and inviting atmosphere to the scene. The video captures the essence of a delicious and indulgent breakfast or brunch dish, with a focus on the textures and colors of the ingredients.', 'video': 'AUZHZ3PH6vM_1_0to102.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'milk', 'distanceToNext': 0.29787678, 'distanceToPrevious': None, 'distanceToQuery': 0.31580538, 'distanceToResult': 0.31865}, {'concept': 'butter', 'distanceToNext': 0.35836744, 'distanceToPrevious': 0.29787678, 'distanceToQuery': 0.33667415, 'distanceToResult': 0.29427707}, {'concept': 'pie', 'distanceToNext': None, 'distanceToPrevious': 0.35836744, 'distanceToQuery': 0.38405937, 'distanceToResult': 0.32269996}]}}, 'caption': 'The video shows a close-up of a dessert being enjoyed. The dessert is a soft serve ice cream cone, which is being dipped into a caramel sauce. The ice cream is white and fluffy, and the caramel sauce is a rich, golden color. The ice cream is being eaten with a spoon, which is visible in the frame. The spoon is being used to scoop up the ice cream and the caramel sauce. The background is blurred, but it appears to be a table with other items on it. The focus of the video is on the ice cream and the caramel sauce, with the spoon and the table serving as secondary elements. The style of the video is casual and informal, with a focus on the enjoyment of the dessert.', 'video': '4tVq3Qvtiz4_55_163to283.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.3397287, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.26698297}, {'concept': 'fresh', 'distanceToNext': 0.3205917, 'distanceToPrevious': 0.3397287, 'distanceToQuery': 0.3249395, 'distanceToResult': 0.24291462}, {'concept': 'tasty', 'distanceToNext': None, 'distanceToPrevious': 0.3205917, 'distanceToQuery': 0.2862119, 'distanceToResult': 0.3310994}]}}, 'caption': 'The video shows a close-up of a dessert being prepared. The dessert appears to be a strawberry shortcake, with layers of cake, whipped cream, and fresh strawberries. The whipped cream is being added to the dessert, and the strawberries are being arranged on top. The dessert is placed on a green plate, which is placed on a table. The background is blurred, but it appears to be a kitchen setting. The style of the video is a slow-motion shot, focusing on the details of the dessert preparation. The colors are vibrant, with the red of the strawberries and the green of the plate contrasting against the white of the whipped cream. The overall impression is one of a delicious and carefully prepared dessert.', 'video': 'GuOEbUnGFAo_10_0to847.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'dessert', 'distanceToNext': 0.19180179, 'distanceToPrevious': None, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.40692067}, {'concept': 'delicious', 'distanceToNext': 0.33099622, 'distanceToPrevious': 0.19180179, 'distanceToQuery': 0.25429422, 'distanceToResult': 0.3267073}, {'concept': 'ingredients', 'distanceToNext': None, 'distanceToPrevious': 0.33099622, 'distanceToQuery': 0.34023887, 'distanceToResult': 0.38424385}]}}, 'caption': "In the video, a person's hand is seen picking up a strawberry from a bowl of whipped cream. The bowl is placed on a table, and there is a container of strawberries nearby. The person's hand is holding the strawberry with the stem, and the strawberry is covered in whipped cream. The video captures the action of the person picking up the strawberry and the contrast between the red strawberry and the white whipped cream. The style of the video is simple and straightforward, focusing on the action of the person picking up the strawberry.", 'video': 'J2Qra6J6t_c_7_0to126.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.2607085, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.2627403}, {'concept': 'taste', 'distanceToNext': None, 'distanceToPrevious': 0.2607085, 'distanceToQuery': 0.2937745, 'distanceToResult': 0.22140312}]}}, 'caption': "The video is a close-up, high-resolution, and colorful food photography sequence. It features a dessert with a creamy, golden-brown topping, possibly a crème brûlée, served in white bowls. The bowls are garnished with fresh berries, including raspberries and blueberries, and a sprig of mint. The bowls are placed on a wooden table, and a silver spoon is used to serve the dessert. The style of the video is elegant and appetizing, with a focus on the textures and colors of the dessert and the fresh fruit garnish. The lighting is soft and warm, highlighting the creamy texture of the dessert and the vibrant colors of the berries. The video is likely intended for a food blog or a cooking channel, showcasing the dessert's presentation and the fresh ingredients used.", 'video': 'GFqdkkxSQ0Q_1_0to105.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'meal', 'distanceToNext': 0.35911697, 'distanceToPrevious': None, 'distanceToQuery': 0.35165465, 'distanceToResult': 0.35108733}, {'concept': 'dessert', 'distanceToNext': 0.43818808, 'distanceToPrevious': 0.35911697, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.27600324}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.43818808, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.37789625}]}}, 'caption': 'The video is a promotional advertisement for a dessert from The Cheesecake Factory. It features a close-up of a warm apple crisp with vanilla ice cream, showcasing the dessert\'s crispy nut topping and the creamy ice cream. The dessert is presented on a white plate with a drizzle of caramel sauce. The background is a gradient of purple and pink, giving the video a soft and inviting feel. The text overlay on the video reads "WARM APPLE CRISP Our Delicious Crispy Nut Topping and Vanilla Ice Cream." The style of the video is simple and appetizing, focusing on the dessert\'s presentation and the enticing description of its ingredients.', 'video': '1p3g2xvjok0_6_0to311.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.2607085, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.27746153}, {'concept': 'taste', 'distanceToNext': 0.10029024, 'distanceToPrevious': 0.2607085, 'distanceToQuery': 0.2937745, 'distanceToResult': 0.24407423}, {'concept': 'tastes', 'distanceToNext': 0.2612626, 'distanceToPrevious': 0.10029024, 'distanceToQuery': 0.36325133, 'distanceToResult': 0.29543382}, {'concept': 'tasty', 'distanceToNext': 0.11825091, 'distanceToPrevious': 0.2612626, 'distanceToQuery': 0.2862119, 'distanceToResult': 0.26651096}, {'concept': 'delicious', 'distanceToNext': 0.2628317, 'distanceToPrevious': 0.11825091, 'distanceToQuery': 0.25429422, 'distanceToResult': 0.19927335}, {'concept': 'fresh', 'distanceToNext': None, 'distanceToPrevious': 0.2628317, 'distanceToQuery': 0.3249395, 'distanceToResult': 0.19661534}]}}, 'caption': 'The video shows a close-up of a plate of food on a wooden table. The food consists of two slices of bread topped with avocado, sliced strawberries, and crumbled feta cheese. The strawberries are bright red and fresh, and the avocado is creamy and green. The feta cheese adds a touch of white to the colorful dish. The plate is white, which contrasts nicely with the vibrant colors of the food. The wooden table provides a warm and rustic backdrop to the meal. The video is likely a food blog or a cooking tutorial, showcasing a healthy and delicious meal.', 'video': '4ou92zGkVyQ_19_405to520.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'tasty', 'distanceToNext': 0.36519802, 'distanceToPrevious': None, 'distanceToQuery': 0.2862119, 'distanceToResult': 0.26969773}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.36519802, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.3549425}]}}, 'caption': 'In the video, a person is seen enjoying a bowl of fruit yogurt. The bowl is pink and filled with a creamy yogurt, topped with blueberries and raspberries. The person is using a spoon to scoop up the yogurt and berries, savoring the sweet and tangy flavors. The setting appears to be a cozy indoor space, with a white chair and a green cushion visible in the background. The overall atmosphere is relaxed and comfortable, as the person indulges in their delicious treat.', 'video': 'Hh3F-GCH7ac_3_18to218.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.4360329, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.28261334}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.4360329, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.35928977}]}}, 'caption': "The video captures the process of creating a chocolate-drizzled whipped cream topping on a dessert. The style of the video is a close-up, slow-motion shot that emphasizes the texture and movement of the chocolate sauce and whipped cream. The chocolate sauce is drizzled from a bottle onto the whipped cream, which is already on top of the dessert. The dessert appears to be a cup of ice cream, as suggested by the presence of the whipped cream and the ice cream cup. The background is blurred, focusing the viewer's attention on the dessert and the chocolate sauce. The overall effect is a visually appealing and appetizing representation of the dessert-making process.", 'video': '6JTizSIQSxg_24_0to160.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'pie', 'distanceToNext': 0.54856884, 'distanceToPrevious': None, 'distanceToQuery': 0.38405937, 'distanceToResult': 0.31009322}, {'concept': 'ingredients', 'distanceToNext': None, 'distanceToPrevious': 0.54856884, 'distanceToQuery': 0.34023887, 'distanceToResult': 0.34071916}]}}, 'caption': 'The video shows a close-up of four mason jars filled with different flavored milkshakes. Each jar is topped with a red and white striped straw and a garnish. The first jar contains a milkshake with a scoop of vanilla ice cream and a slice of bacon on top. The second jar has a chocolate milkshake with a chocolate chip cookie on top. The third jar features a caramel milkshake with a caramel-covered pretzel on top. The fourth jar contains a strawberry milkshake with a strawberry on top. The jars are arranged in a row, and the camera angle is slightly angled, giving a clear view of the milkshakes and their toppings. The style of the video is simple and straightforward, focusing on the presentation of the milkshakes without any additional context or background.', 'video': '6JTizSIQSxg_1_162to294.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'honey', 'distanceToNext': 0.35656774, 'distanceToPrevious': None, 'distanceToQuery': 0.32473093, 'distanceToResult': 0.35562742}, {'concept': 'sweet', 'distanceToNext': 0.46858346, 'distanceToPrevious': 0.35656774, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.2610252}, {'concept': 'ingredients', 'distanceToNext': None, 'distanceToPrevious': 0.46858346, 'distanceToQuery': 0.34023887, 'distanceToResult': 0.3044141}]}}, 'caption': 'The video shows a delicious dessert being prepared and served. In the first frame, a silver bowl is placed on a blue table. The bowl contains a scoop of vanilla ice cream and a chocolate-covered pretzel. In the second frame, a caramel sauce is drizzled over the ice cream and pretzel, adding a rich and sweet flavor to the dessert. In the third frame, the dessert is garnished with a cookie, which is placed on top of the caramel sauce. The cookie adds a crunchy texture to the dessert, complementing the softness of the ice cream and the smoothness of the caramel sauce. The overall style of the video is simple and straightforward, focusing on the dessert and its preparation. The blue table provides a nice contrast to the silver bowl and the colorful dessert, making the dessert stand out. The video does not contain any text or additional elements, keeping the focus solely on the dessert.', 'video': '0J_bFot0a0c_15_0to535.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'tasty', 'distanceToNext': 0.3106761, 'distanceToPrevious': None, 'distanceToQuery': 0.2862119, 'distanceToResult': 0.36202675}, {'concept': 'dessert', 'distanceToNext': 0.49053967, 'distanceToPrevious': 0.3106761, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.30993694}, {'concept': 'freshly', 'distanceToNext': 0.21529531, 'distanceToPrevious': 0.49053967, 'distanceToQuery': 0.42613184, 'distanceToResult': 0.35199898}, {'concept': 'fresh', 'distanceToNext': 0.35226226, 'distanceToPrevious': 0.21529531, 'distanceToQuery': 0.3249395, 'distanceToResult': 0.28697115}, {'concept': 'frozen', 'distanceToNext': None, 'distanceToPrevious': 0.35226226, 'distanceToQuery': 0.39562, 'distanceToResult': 0.4026199}]}}, 'caption': "The video is a demonstration of whipped cream being used in a dessert. The style of the video is a close-up, time-lapse shot focusing on the whipped cream. The first frame shows three containers of whipped cream on a table. The second frame shows the whipped cream being sprayed onto a dessert. The third frame shows the finished dessert with the whipped cream on top. The whipped cream is in three different flavors: vanilla, strawberry, and blueberry. The dessert appears to be a cake or a pie. The whipped cream is being sprayed from a can. The table is wooden and the background is blurred, focusing the viewer's attention on the whipped cream and the dessert.", 'video': '8XhDnG694q4_14_0to208.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.41062486, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.27492142}, {'concept': 'dessert', 'distanceToNext': 0.29179668, 'distanceToPrevious': 0.41062486, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.32203043}, {'concept': 'baked', 'distanceToNext': None, 'distanceToPrevious': 0.29179668, 'distanceToQuery': 0.3318305, 'distanceToResult': 0.25024432}]}}, 'caption': "The video shows a delicious dessert being prepared and served. In the first frame, a white square plate is placed on a wooden table. The plate contains a serving of peach cobbler, which is a golden brown color, indicating it's freshly baked. The cobbler is topped with a scoop of vanilla ice cream.  In the second frame, the ice cream begins to melt, creating a creamy sauce that drizzles over the cobbler. The warm cobbler and cold ice cream create a delightful contrast in temperature and texture.  In the third frame, the ice cream has completely melted, creating a rich, creamy sauce that pools around the cobbler. The sauce enhances the flavor of the peach cobbler, making it even more irresistible. The wooden table provides a rustic backdrop to this mouth-watering dessert.", 'video': 'D7UR9Kbj3Z0_0_0to163.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'sweet', 'distanceToNext': 0.27576005, 'distanceToPrevious': None, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.2872752}, {'concept': 'delicious', 'distanceToNext': None, 'distanceToPrevious': 0.27576005, 'distanceToQuery': 0.25429422, 'distanceToResult': 0.22752029}]}}, 'caption': "The video shows a person eating a chocolate dessert from a white cup. The dessert is a chocolate-covered pastry filled with cream, and it is dusted with powdered sugar. The person is using a spoon to scoop the dessert out of the cup. The dessert is placed on a white plate, which is on a wooden table. The person is holding the spoon over the dessert, and the dessert is being lifted out of the cup. The person is wearing a white shirt, and the spoon is silver. The dessert is being eaten, and the person is enjoying it. The video is a close-up shot of the dessert and the person eating it. The focus is on the dessert and the person's action of eating it. The video is a simple and straightforward depiction of a person enjoying a chocolate dessert.", 'video': '8WwRkvET5jQ_46_0to114.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'snack', 'distanceToNext': 0.3516931, 'distanceToPrevious': None, 'distanceToQuery': 0.28378922, 'distanceToResult': 0.37087017}, {'concept': 'dessert', 'distanceToNext': 0.19180179, 'distanceToPrevious': 0.3516931, 'distanceToQuery': 0.26587576, 'distanceToResult': 0.27292114}, {'concept': 'delicious', 'distanceToNext': 0.11825091, 'distanceToPrevious': 0.19180179, 'distanceToQuery': 0.25429422, 'distanceToResult': 0.22045499}, {'concept': 'tasty', 'distanceToNext': 0.21112806, 'distanceToPrevious': 0.11825091, 'distanceToQuery': 0.2862119, 'distanceToResult': 0.3035413}, {'concept': 'taste', 'distanceToNext': 0.2607085, 'distanceToPrevious': 0.21112806, 'distanceToQuery': 0.2937745, 'distanceToResult': 0.25476807}, {'concept': 'sweet', 'distanceToNext': None, 'distanceToPrevious': 0.2607085, 'distanceToQuery': 0.28567886, 'distanceToResult': 0.2616616}]}}, 'caption': 'The video shows a delightful Easter-themed dessert being prepared. The dessert is a chocolate cupcake nestled in a green paper cup, topped with a generous dollop of green frosting. Adding to the festive touch, the cupcake is adorned with colorful jelly beans and a single white chocolate egg. The video captures the process of assembling these sweet treats, from the initial preparation of the cupcakes to the final decoration with the jelly beans and chocolate egg. The overall style of the video is simple yet charming, focusing on the details of the dessert preparation and the vibrant colors of the ingredients.', 'video': '72B-21bz1TQ_0_0to461.mp4'}, {'_additional': {'semanticPath': {'path': [{'concept': 'juice', 'distanceToNext': 0.56172013, 'distanceToPrevious': None, 'distanceToQuery': 0.35646623, 'distanceToResult': 0.40944242}, {'concept': 'treats', 'distanceToNext': None, 'distanceToPrevious': 0.56172013, 'distanceToQuery': 0.3735571, 'distanceToResult': 0.3323537}]}}, 'caption': 'The video shows a delightful Easter-themed dessert being prepared. The dessert is a chocolate cupcake nestled in a green paper cup, topped with a generous dollop of green frosting. Adding to the festive touch, the cupcake is adorned with colorful jelly beans and a single white chocolate egg. The video captures the process of assembling these sweet treats, from the initial preparation of the cupcakes to the final decoration with the jelly beans and chocolate egg. The overall style of the video is simple yet charming, focusing on the details of the dessert preparation and the vibrant colors of the ingredients.', 'video': '72B-21bz1TQ_0_0to461.mp4'}]}, errors=None)



### 使用查询方法测试


```python
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Move
 

context = client.collections.get("OpenVidContext")
response = context.query.near_text(
    query="food",  
    distance=0.3,
    move_to=Move(force=0.85, concepts=["apples","food"]),
    move_away=Move(force=0.45, concepts=["finance"]),
    return_metadata=wvc.query.MetadataQuery(score=True, certainty=True),
 
)

print(len(response.objects))
for o in response.objects:
    print(o.uuid)
    print(o.properties['caption']) 
    print(o.metadata)

```

    20
    05099ece-04c3-4545-849c-3908be2cc41c
    The video shows a close-up of a plate of food being eaten. The plate contains a fried egg, a slice of ham, and baked beans. The person eating the food is using a fork and knife. The food is being eaten in a casual setting, and the person is enjoying their meal. The video captures the details of the food and the eating process, providing a sense of the meal being enjoyed.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8894022703170776, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    e6e3e0fe-6e87-4718-bb5d-dad5ed5accf0
    The video is a vibrant and colorful display of a refrigerator filled with a variety of food items. The refrigerator is well-stocked with fresh fruits, vegetables, and beverages. The fruits include oranges, bananas, and grapes, while the vegetables include tomatoes and broccoli. The beverages include bottles of juice and water. The refrigerator also contains a variety of desserts, including cupcakes and a jar of berries. The overall style of the video is bright and cheerful, with a focus on the freshness and variety of the food items. The video is likely intended to showcase the abundance and diversity of the refrigerator's contents.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8874735236167908, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    fc91160f-bf4f-4292-bca0-d8fbaa8fd513
    The video captures a close-up view of a plate of food being eaten. The plate contains a colorful dish with various ingredients, including meat, vegetables, and herbs. A spoon is used to scoop up the food, and the person eating the dish is using a knife to cut the meat. The food is being eaten in a restaurant setting, with a tablecloth visible in the background. The style of the video is realistic and it captures the details of the food and the eating process.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8866196870803833, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    fa59f03f-3a6c-486b-a592-b23a54f6aa29
    The video shows a person eating a bowl of soup. The soup contains meat, vegetables, and noodles. The person is using a spoon to eat the soup. The soup is served in a bowl that is placed on a table. The person is wearing a red shirt. The soup appears to be hot and the person is enjoying it. The spoon is made of metal and is being used to scoop up the soup. The vegetables in the soup include carrots and green beans. The meat in the soup is likely chicken or beef. The noodles in the soup are likely egg noodles. The table is made of wood and is brown in color. The person is sitting at the table while eating the soup. The soup is being eaten in a casual setting. The person is likely enjoying a meal at home or in a restaurant. The soup is likely a main course and is being eaten during lunch or dinner. The person is likely using the spoon to eat the soup because it is a convenient utensil for eating soup. The person is likely enjoying the soup because it is a warm and comforting meal. The soup is likely a popular dish in the person's culture or region. The person is likely eating the soup because it is a delicious and satisfying meal. The soup is likely a traditional dish that has been passed down through generations. The person is likely eating the soup because it is a nutritious and healthy meal. The soup is likely a popular dish in the person's community or social circle. The
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8866066932678223, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    c2e882f8-28b2-49a3-82ee-686d5f18d927
    The video is a compilation of three different meals, each presented in a metal container. The meals are diverse and colorful, featuring a variety of fruits, vegetables, and salads. The first meal includes a mix of fruits such as bananas, strawberries, and blueberries, as well as a salad with lettuce and carrots. The second meal consists of a salad with tomatoes and cucumbers, along with a side of crackers. The third meal features a salad with lettuce, tomatoes, and carrots, accompanied by a side of fruit. The style of the video is simple and straightforward, focusing on the presentation of the meals without any additional context or narrative. The meals are arranged neatly in the containers, and the colors of the fruits and vegetables are vibrant and appealing. The overall impression is one of healthy and nutritious eating.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8864352703094482, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    82c49e0f-03a0-445b-8b00-d0030ec16663
    The video captures a person enjoying a meal at a street food stall. The person is using chopsticks to eat from a bowl of soup, which contains noodles, vegetables, and meat. The soup is served in a yellow bowl, and there is a small bowl of seasoning nearby. The table is covered with a white tablecloth, and there are other dishes and condiments in the background. The setting suggests a casual and relaxed dining experience, with the focus on the delicious food being enjoyed.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8861154317855835, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    aaa8a141-3f54-4607-8fb8-f578c06d7e53
    The video captures a vibrant street food scene. The first frame shows a variety of fresh vegetables and fruits on display, including green and yellow vegetables, and a selection of fruits. The second frame introduces a variety of cooked dishes, including a bowl of red sauce and a tray of cooked meat. The third frame showcases a variety of cooked dishes, including a tray of cooked meat and a bowl of red sauce. The style of the video is a close-up shot of the food, focusing on the colors and textures of the ingredients. The video is likely to be a food documentary or a cooking show, showcasing the variety and freshness of the ingredients used in street food.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.884285569190979, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    987fd10d-9d0f-44b8-9762-ab90b55b2654
    The video shows a close-up of a plate of food being eaten with a fork and knife. The food consists of a piece of fish, possibly a whole fish, cooked and served in a sauce or broth. The fish is garnished with vegetables, including carrots and possibly other colorful vegetables. The sauce or broth is yellowish in color, suggesting it might be a curry or a similar type of sauce. The fork and knife are being used to cut and eat the fish, indicating that the person eating the food is in the process of enjoying their meal. The style of the video is a simple, straightforward food video, focusing on the food and the act of eating it. The video does not contain any additional elements or background, keeping the focus solely on the plate of food and the utensils being used to eat it.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.884240984916687, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    8a38ea40-83fb-40b1-b0b7-f60790e7ac6d
    The video shows a fish being cooked in a pot with various vegetables. The fish is placed in the pot and covered with a variety of colorful vegetables, including tomatoes, mushrooms, and green herbs. The pot is filled with a clear broth, and the fish is cooked until it is fully cooked and the vegetables are tender. The video captures the process of cooking the fish and the vibrant colors of the vegetables. The style of the video is a close-up shot of the cooking process, focusing on the fish and the vegetables. The video is likely to be used for cooking tutorials or food blogs.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8841015100479126, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    dc16d27b-fbfc-4009-9b7c-5551bed93860
    The video captures a delicious meal being served on a wooden table. The meal consists of a variety of meats, including sausage, chicken, and beef, all cooked to perfection. The meats are accompanied by a side of crispy french fries and a fresh salad, adding a touch of green to the plate. A bowl of brown sauce is also present, ready to enhance the flavors of the meal. The arrangement of the food on the wooden table creates an inviting and appetizing scene. The video is likely a food review or a cooking tutorial, showcasing the preparation and presentation of a hearty and satisfying meal.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8835453391075134, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    d15817a1-e075-4b4f-b175-3f802cc10eb1
    The video shows a close-up of a bowl of beef noodle soup being eaten with chopsticks. The soup contains thin white noodles, slices of beef, and green vegetables. The chopsticks are used to pick up a piece of beef and a noodle. The soup is served in a pink bowl, and the spoon is placed on the side of the bowl. The video captures the process of eating the soup, from the initial picking up of the food to the final consumption. The focus is on the food and the eating process, with no other objects or people in the frame. The style of the video is simple and straightforward, focusing on the food and the act of eating.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.883479118347168, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    3af0796a-0334-46ef-b240-ae6690b64ead
    The video captures a close-up view of a meal being enjoyed with chopsticks. The meal consists of a bowl of soup filled with noodles, meat, and vegetables. The chopsticks are used to pick up a piece of meat, which is then dipped into the soup. The soup is rich and flavorful, with the noodles, meat, and vegetables all contributing to the overall taste. The chopsticks are used skillfully, demonstrating the art of eating with chopsticks. The meal appears to be a traditional Asian dish, possibly a noodle soup. The close-up view allows for a detailed examination of the ingredients and the technique used to eat the dish.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8833267688751221, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    5eeb883f-f5e3-4291-bfd8-d8400dad9bc2
    The video shows a plate of food being eaten. The plate contains a variety of seafood, including fish and shrimp, as well as some fries. The food is being eaten with a fork and knife. The person eating the food is not visible in the video. The style of the video is a simple, straightforward documentation of the food being eaten. There are no special effects or artistic elements. The focus is solely on the food and the act of eating it.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8833185434341431, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    9f04a1fc-e387-4cb4-8dfa-ce9dd28d8508
    The video captures a meal in progress, with a focus on the table setting and the food being consumed. The table is set with a white tablecloth and a placemat, and there are two glasses of water and one glass of orange juice. The food consists of a sandwich, a salad, and a pastry, all served on a white plate. The sandwich appears to be made with bread, cheese, and vegetables, while the salad includes tomatoes and cucumbers. The pastry is a quiche, filled with cheese and vegetables. The meal is being enjoyed in a relaxed and leisurely manner, with the person eating taking their time to savor each bite. The overall style of the video is casual and intimate, capturing the simple pleasure of a good meal.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8832565546035767, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    dcbd1c43-c749-469d-bdcd-522f2354ea2d
    The video shows a close-up of a meal being eaten with chopsticks. The meal consists of a white bowl filled with a salad and a piece of fried food, possibly chicken or fish. The salad is a mix of shredded vegetables, including carrots and cabbage, and is garnished with a green leafy vegetable. The fried food is golden brown and appears to be crispy. The chopsticks are being used to pick up a piece of the fried food, which is being lifted out of the bowl. The style of the video is a simple, straightforward food video, focusing on the meal and the act of eating it. The lighting is bright and even, highlighting the colors and textures of the food. The video does not contain any text or additional elements, keeping the focus solely on the meal and the chopsticks.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8822977542877197, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    fa0a1836-3c6b-434e-bae5-72c40866615a
    The video is a culinary journey, showcasing a meal being prepared and served. The style is a close-up, high-definition food photography, focusing on the textures and colors of the ingredients. The meal consists of a soup, a salad, and a bread dish. The soup is a rich, orange-colored broth, filled with chunks of meat and vegetables, and garnished with fresh herbs and a slice of tomato. The salad is a vibrant mix of green leaves, purple onions, and red tomatoes, with a sprinkle of nuts on top. The bread dish features a loaf of bread, sliced and ready to be served. The video captures the process of preparing the meal, from chopping the vegetables to serving the dishes. The overall style is clean and simple, emphasizing the natural beauty of the ingredients and the careful preparation of the meal.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8812576532363892, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    1998a945-2197-4fca-b751-1379c6a27bc8
    The video is a culinary journey, showcasing a meal being prepared and served. The style is a close-up, high-definition food photography, focusing on the textures and colors of the ingredients. The meal consists of a soup, a salad, and a bread dish. The soup is a rich, orange-colored broth, filled with chunks of meat and vegetables, and garnished with fresh herbs and a slice of tomato. The salad is a vibrant mix of green leaves, purple onions, and red tomatoes, with a sprinkle of nuts on top. The bread dish features a loaf of bread, sliced and ready to be served. The video captures the process of preparing the meal, from chopping the vegetables to serving the dishes. The overall style is clean and simple, emphasizing the natural beauty of the ingredients and the careful preparation of the meal.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8812576532363892, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    01ad7b68-2a79-4ed0-bc20-883db28212be
    The video shows a meal being enjoyed at a restaurant. The meal consists of two plates of food, each containing a variety of dishes. The plates are placed on a wooden table, and there are utensils such as a spoon and a fork visible. The food appears to be a mix of Asian cuisine, with dishes like rice, meat, and vegetables. The setting suggests a casual dining experience, and the meal seems to be in progress, with some of the food already consumed. The lighting in the restaurant is warm and inviting, creating a cozy atmosphere.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8812175393104553, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    773a54cf-89eb-4635-a965-735f2c5343eb
    The video shows a close-up view of a salad bar with various food items and condiments. The salad bar is filled with a variety of fresh vegetables, including tomatoes, onions, cucumbers, and lettuce. There are also several bottles of condiments, such as ranch dressing, honey mustard, and barbecue sauce. The salad bar is well-stocked and appears to be ready for customers to serve themselves. The style of the video is straightforward and informative, focusing on the food items and condiments available at the salad bar. The video does not contain any people or animals, and the focus is solely on the food items and condiments. The video is likely intended to showcase the variety of options available at the salad bar, and to encourage customers to try the different items.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8809212446212769, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    4d6a313d-027d-4944-9648-e7212bf76f36
    The video shows a close-up of a bowl of food, likely a soup or stew, being eaten by a person. The bowl is filled with a creamy white liquid, possibly a broth or sauce, and contains chunks of meat and vegetables. The person is using chopsticks to pick up the food, indicating that the dish may be of Asian origin. The food appears to be well-prepared and appetizing, with the meat and vegetables cooked to perfection. The person is enjoying the meal, taking their time to savor each bite. The overall style of the video is simple and straightforward, focusing on the food and the person's enjoyment of it.
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=0.8803462386131287, score=0.0, explain_score=None, is_consistent=None, rerank_score=None)
    

#### 可度量的混合搜索

度量方式 
结果集融合方法和数据源权重



```python
from weaviate.classes.query import MetadataQuery

vector_names = ["Caption"]
response = jeopardy.query.hybrid(
    query="A Ice cream appears in front of the window, and the view outside the window is very peaceful.",  
    target_vector=vector_names,  # Specify the target vector for named vector collections
    limit=3,
    alpha=0.1,
    query_properties=["caption"],  
    return_metadata=MetadataQuery(score=True, explain_score=True, distance=True),
)



for o in response.objects:
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata)
```

    The video captures a man's reaction to a purple sports car. The man, dressed in a black shirt, is seen in the foreground, his face lit up with excitement as he gestures towards the car. The car, a striking shade of purple, is parked in the background, its doors open, inviting viewers to take a closer look. The setting appears to be a gas station, with other cars and trucks visible in the background. The overall style of the video suggests a casual, spontaneous moment captured on camera, with the man's enthusiasm for the car being the main focus.
    static
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.0, explain_score='\nHybrid (Result Set vector,hybridVector) Document ff7101d9-a55f-455f-a51a-3d8330e6264c: original score -0.040314913, normalized score: 1', is_consistent=None, rerank_score=None)
    The video is a 3D animated scene featuring a group of yellow rubber ducks floating in a body of water. The ducks are arranged in a line, with one duck slightly ahead of the others. The water is calm and reflects the ducks and the surrounding environment. In the background, there are large rocks and trees, creating a natural setting. The scene is brightly lit, suggesting it is daytime. The overall style of the video is playful and whimsical, with a focus on the ducks and their journey through the water.
    pan_left
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.0, explain_score='\nHybrid (Result Set vector,hybridVector) Document fe293f6c-4575-4ab8-8dad-8b312490054f: original score -0.040314913, normalized score: 1', is_consistent=None, rerank_score=None)
    The video features a 3D animated character, a young girl with blonde hair, wearing a purple hooded cloak and pink shoes. She is holding a black pot with a brown lid. The girl is walking on a wooden floor with a grid pattern, and the background is blurred, suggesting movement. The style of the animation is cartoonish and colorful, with a focus on the character's facial expressions and the texture of the wooden floor. The overall mood of the video is cheerful and playful.
    zoom_in+pan_right
    MetadataReturn(creation_time=None, last_update_time=None, distance=None, certainty=None, score=1.0, explain_score='\nHybrid (Result Set vector,hybridVector) Document fc32b93c-50c4-4fe1-8380-3b0ca58ffcf3: original score -0.040314913, normalized score: 1', is_consistent=None, rerank_score=None)
    


```python
from weaviate.classes.query import MetadataQuery, Rerank

vector_names = ["Caption"]
response = jeopardy.query.bm25(
    query="food outside",   
    limit=100,  
    query_properties=["caption"],  
    rerank=Rerank(
        prop="caption",
        query="food, room, outside"
    ),
    return_metadata=MetadataQuery(distance=True, explain_score=True,score=True),
)



for o in response.objects:
    print(o.uuid)
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata.score, o.metadata.explain_score)
```

    d96a018a-9e5b-4574-add8-8a385d8c65e5
    The video shows a person in a red bracelet reaching for food in a buffet line. The food includes a variety of dishes such as sandwiches, salads, and seafood. The person is using a serving spoon to pick up the food. The buffet line is filled with plates and bowls of different foods. The person is wearing a red bracelet on their wrist. The food is displayed on a glass shelf. The person is standing in front of the glass shelf. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet. The person is picking up food from the glass shelf. The person is wearing a red bracelet.
    tilt_up+pan_left
    2.5725202560424805 , BM25F_food_frequency:15, BM25F_food_propLength:47
    e87a96c4-7c08-4202-b9f4-81f07c93247d
    In the video, a woman is seen eating a piece of food on a plate. She is standing in front of a television screen that displays a cityscape. The woman is wearing a blue dress and has a plate of food in her hand. She is eating the food with her mouth open. The television screen is turned on and is showing a cityscape. The woman is standing in front of the television screen. The woman is eating the food. The television screen is turned on. The woman is standing. The television screen is displaying a cityscape. The woman is wearing a blue dress. The woman is holding a plate of food. The woman is eating the food. The television screen is turned on. The woman is standing in front of the television screen. The woman is eating the food. The television screen is displaying a cityscape. The woman is wearing a blue dress. The woman is holding a plate of food. The woman is eating the food. The television screen is turned on. The woman is standing in front of the television screen. The woman is eating the food. The television screen is displaying a cityscape. The woman is wearing a blue dress. The woman is holding a plate of food. The woman is eating the food. The television screen is turned on. The woman is standing in front of the television screen. The woman is eating the food. The television screen is displaying a cityscape. The woman is wearing a blue dress. The woman is holding a plate
    static
    2.5813469886779785 , BM25F_food_frequency:13, BM25F_food_propLength:35
    79383c4e-bb95-40cf-80a3-16ada33892bd
    The video shows a person eating a plate of rice with meat and sauce. The person is using a fork to pick up the food. The plate is white and the food is colorful. There is a small bowl of sauce on the side of the plate. The person is sitting at a table. The table is red. The person is wearing a white shirt. The person is holding the fork in their right hand. The person is eating the food. The food looks delicious. The person is enjoying their meal. The person is eating the food. The food is on the plate. The plate is on the table. The table is red. The person is sitting at the table. The person is wearing a white shirt. The person is holding the fork in their right hand. The person is eating the food. The food looks delicious. The person is enjoying their meal. The person is eating the food. The food is on the plate. The plate is on the table. The table is red. The person is sitting at the table. The person is wearing a white shirt. The person is holding the fork in their right hand. The person is eating the food. The food looks delicious. The person is enjoying their meal. The person is eating the food. The food is on the plate. The plate is on the table. The table is red. The person is sitting at the table. The person is wearing a white shirt. The person is holding the fork in their right hand.
    static
    2.5739798545837402 , BM25F_food_frequency:14, BM25F_food_propLength:42
    4df61bad-bd05-45a1-b4e2-7fed88472cc9
    The video shows a cozy and modern living space. In the first frame, a black leather bench is positioned against a white wall with a geometric pattern. A wooden table is placed in front of the bench, and a white coffee cup is placed on a small plate. A colorful throw pillow is placed on the bench, adding a pop of color to the space.  In the second frame, the camera angle changes to show a window with a view of a lush green forest. The window is framed with black trim, and the view outside is serene and peaceful.  In the third frame, the camera angle changes again to show a black leather armchair with a throw pillow. A wooden side table is placed next to the armchair, and a small bowl of fruit is placed on the table. The armchair is positioned in front of the window, providing a comfortable spot to relax and enjoy the view.  The style of the video is modern and minimalist, with a focus on clean lines and neutral colors. The use of natural light and the view of the forest outside adds a touch of warmth and tranquility to the space. The video captures the essence of a modern and comfortable living space, with a focus on functionality and style.
    tilt_down
    2.510058641433716 , BM25F_outside_frequency:2, BM25F_outside_propLength:84
    ab7e8bbe-50bc-4ae3-b5de-91ff9f64a3c4
    The video shows a close-up view of a car's wheel and brake system. The wheel is black with a silver center and a red brake caliper. The brake caliper is visible in the center of the wheel. The car's logo is visible on the center of the wheel. The wheel is made of metal and has a shiny finish. The brake caliper is made of metal and has a red finish. The brake caliper is located on the inside of the wheel. The wheel is attached to the car's axle. The car's axle is made of metal and has a black finish. The car's axle is located on the inside of the wheel. The car's axle is attached to the car's frame. The car's frame is made of metal and has a black finish. The car's frame is located on the inside of the wheel. The car's frame is attached to the car's body. The car's body is made of metal and has a black finish. The car's body is located on the outside of the wheel. The car's body is attached to the car's frame. The car's frame is attached to the car's axle. The car's axle is attached to the car's wheel. The car's wheel is attached to the car's frame. The car's frame is attached to the car's
    pan_right
    2.391245126724243 , BM25F_outside_frequency:1, BM25F_outside_propLength:38
    828e1562-1f7c-4d3e-864b-d41c1be5af73
    The video captures a close-up view of a plate of food, showcasing a variety of dishes in a visually appealing manner. The plate is filled with a mix of rice, vegetables, and meat, all arranged in an appetizing manner. The colors of the food are vibrant, with the green of the vegetables contrasting against the white of the rice and the brown of the meat. The food is garnished with a sprig of orange and green leaves, adding a touch of freshness to the dish. The plate is placed on a wooden table, which provides a warm and rustic backdrop to the food. The overall style of the video is that of a food blog or a cooking show, where the focus is on the presentation and the visual appeal of the food. The video does not contain any text or narration, allowing the viewer to focus solely on the food. The video is shot in a way that highlights the textures and colors of the food, making it look even more delicious. The close-up shots of the food give the viewer a detailed view of the ingredients and the preparation, making it an informative and enjoyable watch.
    static
    2.303182363510132 , BM25F_food_propLength:94, BM25F_food_frequency:9
    de8a3bcf-85f5-4ae7-9ddd-2659ed63fdc2
    The video shows a man in a kitchen preparing a meal. He is standing at a counter with a cutting board and a pan. He is cutting meat on the cutting board. The kitchen is well-equipped with various appliances and utensils. The man is focused on his task, and the kitchen appears to be clean and organized. The lighting in the kitchen is bright, and the colors are warm. The man is wearing a gray shirt, and he has a beard. The kitchen has wooden cabinets and a stainless steel stove. The man is using a knife to cut the meat, and he is holding a piece of meat in his hand. The counter is made of granite, and there are various bottles and bowls on it. The man is looking at the meat he is cutting, and he appears to be concentrating on his task. The kitchen has a window, and there is a view of the outside. The man is the only person in the kitchen, and he is the main focus of the video. The video is likely a cooking tutorial or a food show.
    static
    2.9263417720794678 , BM25F_food_frequency:1, BM25F_food_propLength:77, BM25F_outside_frequency:1, BM25F_outside_propLength:77
    88478ab5-a754-4d7f-9e7b-aafff56d88ce
    The video shows a white camper van parked on a grassy area. In the first frame, the van is stationary, and there are no chairs or tables outside. In the second frame, two black chairs and a small table have been set up outside the van. In the third frame, the chairs and table are still outside the van, but there are also two cups on the table. The style of the video is a simple, straightforward documentation of the setup of the chairs and table outside the camper van. The focus is on the van and the outdoor furniture, with no additional action or movement. The video does not include any people or animals, and the background is a simple grassy area with no other objects or landmarks. The lighting in the video is natural, suggesting that the video was taken during the day. The overall style of the video is minimalistic and functional, focusing on the practical aspects of setting up outdoor furniture for use with a camper van.
    tilt_up+pan_right
    3.2749814987182617 , BM25F_outside_frequency:4, BM25F_outside_propLength:78
    16f51d97-e1cf-4d01-83c4-88e8ed49e029
    The video is a close-up of a plate of food, showcasing a gourmet meal. The plate is white, and the food consists of a dark meat, possibly beef or lamb, accompanied by a creamy sauce. The meat is cooked to perfection, with a rich, dark color that suggests it's been cooked for a long time. The sauce is light and creamy, adding a touch of elegance to the dish. The food is garnished with a dollop of sour cream and a sprig of parsley, adding a pop of color and a hint of freshness to the dish. The overall style of the video is simple and elegant, focusing on the food and the details of the dish. The camera angle is close-up, allowing the viewer to appreciate the textures and colors of the food. The lighting is soft and warm, highlighting the colors of the food and creating a cozy atmosphere. The video does not contain any text or additional elements, keeping the focus solely on the food.
    Undetermined
    2.2407050132751465 , BM25F_food_frequency:7, BM25F_food_propLength:84
    a580d0e4-534d-47f9-9fba-d8fe3aaf6801
    In the video, a man is seen drinking from a red cup. He is wearing a black hoodie with a yellow logo on the left side. The man is standing in a room with a window in the background. The window has a white frame and a small tree can be seen outside. The man is holding the cup with both hands and appears to be enjoying his drink. The room has a simple and clean design, with the window being the main focus. The man's hoodie adds a pop of color to the otherwise neutral tones of the room. The yellow logo on his hoodie stands out against the black fabric. The small tree outside the window adds a touch of nature to the urban setting. The man's action of drinking from the cup suggests that he is taking a break or enjoying a moment of relaxation. The overall style of the video is casual and relaxed, with a focus on the man and his drink.
    static
    2.598611831665039 , BM25F_outside_frequency:2, BM25F_outside_propLength:76
    f06437e5-9167-4b1f-b7e4-d6e4f7d58070
    The video shows a close-up of a metal tray filled with golden-brown, crispy potato wedges. The wedges are arranged in a pile, with some overlapping each other. The tray is placed on a dark surface, which contrasts with the bright color of the wedges. The wedges appear to be freshly cooked, with a golden-brown color indicating they are crispy on the outside. The style of the video is simple and straightforward, focusing solely on the potato wedges without any additional elements or distractions. The video does not contain any text or other objects, and the focus is solely on the potato wedges and the tray they are placed on. The video does not contain any action or movement, and the wedges remain stationary throughout the video. The video is likely intended to showcase the potato wedges, possibly for a recipe or a food review.
    static
    3.0194108486175537 , BM25F_outside_propLength:72, BM25F_food_frequency:1, BM25F_food_propLength:72, BM25F_outside_frequency:1
    453d283d-45b8-488e-8658-7e3037d0c5dd
    The video is a dynamic and colorful journey through a mountainous landscape, viewed from the interior of a luxury sports car. The car's interior is sleek and modern, with a black and orange color scheme, featuring a large touch screen display, a digital instrument cluster, and a variety of controls and buttons. The car's interior is well-lit, with the sunroof open, allowing natural light to flood in. The car is in motion, as evidenced by the blurred scenery outside the car's windows. The scenery outside the car is breathtaking, with rugged mountains and a winding road that the car is traveling on. The car's speed and the road's curves create a sense of motion and excitement. The overall style of the video is dynamic and engaging, capturing the thrill of driving a luxury sports car through a stunning natural landscape.
    static
    2.6101222038269043 , BM25F_outside_frequency:2, BM25F_outside_propLength:75
    7086c2ad-f4fc-4ef2-97d2-65bb8c3aa848
    The video shows a close-up of a plate of food being prepared. The plate is white with a blue rim and a floral design on the side. On the plate, there is a serving of fried rice with green peas mixed in. Next to the rice, there is a fried egg with the yolk intact. The egg is golden brown and appears to be cooked sunny-side up. Accompanying the egg and rice are slices of cucumber and tomato, which add a fresh element to the dish. The food is presented on a wooden table, and the background is blurred, focusing the viewer's attention on the plate of food. The style of the video is simple and straightforward, with a focus on the food being prepared. The colors are vibrant, and the textures of the food are clearly visible. The video does not contain any text or additional elements, keeping the focus solely on the food.
    static
    2.200500011444092 , BM25F_food_frequency:6, BM25F_food_propLength:78
    e852dd12-ebc2-4fab-b54f-07838e66c552
    The video captures a man's journey from the outside to the inside of a race car. In the first frame, the man is standing outside the car, looking at it with anticipation. In the second frame, he is in the process of getting into the car, his hands gripping the door handle. In the third frame, he is fully inside the car, sitting in the driver's seat with a look of excitement and determination on his face. The car itself is sleek and modern, with a black exterior and a white interior. The man is dressed casually, wearing a gray shirt and black pants. The video is shot from a low angle, emphasizing the height of the car and the man's determination to climb inside. The overall style of the video is dynamic and exciting, capturing the anticipation and excitement of the man as he prepares to drive the race car.
    static
    2.645273447036743 , BM25F_outside_frequency:2, BM25F_outside_propLength:72
    1fb02e95-7428-4c60-b414-ba43fa8efd88
    The video shows a close-up of a plate of food being eaten. The plate contains a variety of foods, including meat, potatoes, mushrooms, sauerkraut, and corn. The food is being eaten with a fork and knife. The style of the video is a simple, straightforward food video, with no additional elements or embellishments. The focus is solely on the food and the act of eating it. The video does not contain any text or additional graphics. The camera angle is a close-up shot of the plate, providing a detailed view of the food. The lighting is bright, highlighting the colors and textures of the food. The video does not contain any sound or music. The overall style of the video is simple and straightforward, focusing on the food and the act of eating it.
    static
    2.3405683040618896 , BM25F_food_frequency:7, BM25F_food_propLength:60
    5c3fde89-a325-44f8-810e-1670cf986ba4
    The video captures a person's hand holding a bowl of food, which is filled with a variety of colorful ingredients. The bowl is placed on a wooden table, and the hand is holding the bowl from the side. The food in the bowl includes a mix of vegetables, such as green onions and red peppers, as well as some meat. The colors of the ingredients are vibrant, and the arrangement of the food in the bowl is visually appealing. The wooden table provides a natural and rustic backdrop for the bowl of food. The overall style of the video is simple and straightforward, focusing on the food and the hand holding the bowl. The video does not contain any text or additional elements, and the focus is solely on the food and the hand holding the bowl.
    static
    2.2650907039642334 , BM25F_food_propLength:64, BM25F_food_frequency:6
    a26ae289-99f6-4de9-a42d-8685f95d1937
    The video shows a close-up of a plate of food, featuring a large ball of cream cheese topped with sliced almonds, surrounded by a variety of crackers and slices of bread. The food is presented on a wooden plate, which is placed on a table. The style of the video is a simple, straightforward food presentation, with no additional elements or actions. The focus is solely on the food, showcasing its textures and colors. The lighting is bright, highlighting the details of the food items. The video does not contain any text or additional elements, and the camera angle remains constant, providing a clear view of the food. The overall style of the video is minimalistic and straightforward, focusing solely on the presentation of the food.
    static
    2.3105337619781494 , BM25F_food_propLength:67, BM25F_food_frequency:7
    ab1dd2c0-2e1b-4f64-9a86-b0b073383bc7
    The video shows a close-up of a plate of food being eaten with a fork. The food consists of shrimp, tomatoes, and feta cheese, all mixed together in a sauce. The fork is being used to pick up a piece of the shrimp, which is covered in the sauce and garnished with a sprig of parsley. The plate is white with orange stripes around the edge, and it's placed on a blue and white checkered tablecloth. The style of the video is a simple, straightforward food video, focusing on the textures and colors of the food. The lighting is bright and even, highlighting the details of the food and the plate. The video does not contain any text or additional elements, and it's shot in a way that emphasizes the food and the eating experience.
    Undetermined
    2.232328414916992 , BM25F_food_frequency:6, BM25F_food_propLength:71
    ff140df3-2191-4901-a86f-2652a90cfa72
    The video shows a close-up of a white plate filled with a green and brown food item, which appears to be a salad or a similar dish. The food is piled high on the plate, with various textures and colors visible, suggesting a mix of ingredients. The plate is placed on a white countertop, which provides a neutral background that contrasts with the vibrant colors of the food. The style of the video is simple and straightforward, focusing on the food without any additional context or embellishments. The lighting is bright and even, highlighting the textures and colors of the food without creating harsh shadows or glare. The video does not contain any text or additional elements, keeping the focus solely on the plate of food.
    static
    2.2462525367736816 , BM25F_food_frequency:6, BM25F_food_propLength:68
    2b8688c4-74b9-4ae9-8658-0c56db95162b
    The video shows a close-up of a wooden table with a yellow bowl containing a creamy white substance, possibly a sauce or soup, and a whole green artichoke. The artichoke is placed to the left of the bowl, and the bowl is positioned towards the center of the table. The table appears to be a kitchen countertop, and the focus is on the food items, suggesting a cooking or food preparation scene. The style of the video is simple and straightforward, with a focus on the food items and the table setting. The lighting is bright and even, highlighting the textures and colors of the food and the table. The video does not contain any text or additional elements, and the focus is solely on the food and the table setting.
    pan_left
    2.204393148422241 , BM25F_food_frequency:5, BM25F_food_propLength:61
    6d7fddac-f832-4fc1-bdfc-1980790f81e1
    The video features a man standing in front of a large window, looking out at a crowd of people gathered in a field. He is wearing a black jacket and a blue shirt, and his mouth is slightly open as if he is speaking or about to speak. The window behind him is wide and clear, allowing a view of the outdoor scene. The man appears to be indoors, possibly in a building or a room with a large window. The crowd outside is not clearly visible, but their presence is suggested by the man's gaze and the context of the scene. The style of the video is realistic and naturalistic, with no apparent filters or artistic effects applied. The focus is on the man and his interaction with the scene outside the window.
    static
    2.598611831665039 , BM25F_outside_frequency:2, BM25F_outside_propLength:76
    70e6a508-9b6a-45a8-ba6a-0df06ed55661
    The video captures a close-up view of a plate of food, featuring sausage and potatoes. The sausage is browned and appears to be cooked, while the potatoes are golden and seasoned with herbs. The food is arranged on a white plate, which contrasts with the colors of the food. The style of the video is simple and straightforward, focusing on the food without any additional elements or distractions. The camera angle is slightly above the plate, providing a clear view of the food. The lighting is bright, highlighting the textures and colors of the food. The video does not contain any text or additional elements, focusing solely on the food. The overall impression is of a simple, yet delicious meal.
    pan_right
    2.3319075107574463 , BM25F_food_frequency:7, BM25F_food_propLength:62
    97a4c34e-5a6e-4faa-a6c5-dc7f24988483
    The video is a short, simple, and appetizing clip featuring a plate of fried food, likely chicken, accompanied by a bowl of ketchup. The food is golden brown and appears crispy, while the ketchup is a vibrant red. The plate is white, which contrasts nicely with the food and the ketchup. The background is blurred, but it seems to be a green, leafy environment, possibly a garden or a park. The style of the video is straightforward and focuses on the food, with no additional elements or distractions. The lighting is bright and even, highlighting the textures and colors of the food and the ketchup. The video is likely intended to showcase the food's appeal and to encourage viewers to try it.
    pan_right
    2.2509326934814453 , BM25F_food_frequency:6, BM25F_food_propLength:67
    325cd557-f4d7-42c9-b4bb-4cda088bbd97
    The video shows a street food vendor's stall with a variety of food items displayed. The vendor has a large metal pole with several bags of food hanging from it, and a blue basket filled with more food items. The food appears to be wrapped in leaves, suggesting it might be a type of traditional or ethnic cuisine. The vendor's stall is set up on a wooden table, and there are other items in the background, including a metal pot and a bottle. The style of the video is a simple, straightforward documentation of the food vendor's stall, with no additional embellishments or artistic effects. The focus is on the food and the setting, providing a clear view of the items available for purchase.
    static
    2.293714761734009 , BM25F_food_frequency:7, BM25F_food_propLength:71
    4b441dc5-72a7-4e06-9329-2bdaae72dfa1
    The video shows a person holding a white plate with a serving of food. The food consists of two meat patties, possibly beef, and several boiled potatoes. The potatoes are scattered around the meat patties, and the dish is covered in a light brown sauce. The person is holding the plate with both hands, and the background is blurred, focusing the viewer's attention on the plate of food. The style of the video is simple and straightforward, with a focus on the food and the person holding the plate. The lighting is bright, and the colors are vibrant, making the food look appetizing. The video does not contain any text or additional elements, and the focus is solely on the plate of food.
    static
    2.279428243637085 , BM25F_food_frequency:6, BM25F_food_propLength:61
    e2939dd3-b9c7-45a4-b511-fda4166dad48
    The video shows a close-up of a plate of food being served. The plate contains a variety of items, including a salad with lettuce and carrots, a piece of grilled tofu, and a side of rice. The tofu is topped with a sauce, and there are also some pickles on the plate. The food is presented on a white plate, which is placed on a table. The style of the video is simple and straightforward, focusing on the food and the presentation. The camera angle is close-up, allowing for a detailed view of the food. The lighting is bright, highlighting the colors and textures of the food. The video does not contain any text or additional elements. The focus is solely on the food and the presentation.
    static
    2.2603516578674316 , BM25F_food_frequency:6, BM25F_food_propLength:65
    ecb6051c-1710-4794-b849-6f0cbe84192e
    The video shows a close-up of a bowl of food being eaten. The food consists of a mix of ingredients, including meat, onions, and green peas. The dish is covered in a red sauce, and the person eating the food is using a spoon to scoop up the ingredients. The style of the video is a simple, straightforward food video, focusing on the textures and colors of the ingredients. The camera is close to the bowl, capturing the details of the food and the person's interaction with it. The lighting is bright, highlighting the vibrant colors of the ingredients and the red sauce. The video does not contain any text or additional elements, focusing solely on the food and the person eating it.
    static
    2.2890875339508057 , BM25F_food_frequency:6, BM25F_food_propLength:59
    49b07957-a876-4359-8233-77c2cda759f0
    The video shows a person using a food processor on a wooden table. The food processor is black and silver, with a digital display and buttons on the front. In the first frame, the food processor is empty. In the second frame, the person has added ingredients to the food processor. In the third frame, the person is operating the food processor, and the ingredients are being processed. The style of the video is a simple, straightforward demonstration of how to use a food processor. The focus is on the food processor and the person's hands, with no additional elements or distractions. The video is likely intended for instructional purposes, such as a tutorial or a recipe video.
    static
    2.3492934703826904 , BM25F_food_frequency:7, BM25F_food_propLength:58
    f8d72eab-190f-49f2-8410-94309faa973b
    The video features a man in a blue shirt and a white apron with a logo on it, presenting a plate of food. He is standing in front of a white wall, and the lighting is bright. The man is holding the plate with both hands, and he appears to be speaking or explaining something about the food. The food on the plate consists of several small, round, golden-brown items that could be fried or baked. The style of the video is casual and informal, with a focus on the food being presented. The man's attire and the logo on his apron suggest that he may be a chef or a food enthusiast. The overall impression is that of a friendly and approachable cooking or food-related video.
    static
    2.2509326934814453 , BM25F_food_frequency:6, BM25F_food_propLength:67
    128b4ac8-7aee-4ff3-9a3f-3e1a5b7ef696
    In the video, a person is seen frying food in a large pan. The food appears to be small, round, and golden brown, suggesting that they are being deep-fried. The person is using a large strainer to remove the food from the pan, indicating that they are in the process of cooking or serving the food. The pan is placed on a stove, which is visible in the background. The person is wearing a red apron, suggesting that they are in a kitchen or a cooking area. The overall style of the video is a simple, straightforward depiction of a cooking process, with no additional elements or distractions. The focus is solely on the person, the pan, and the food being cooked.
    static
    2.209804058074951 , BM25F_food_frequency:5, BM25F_food_propLength:60
    2a7a4ffb-11c9-4d49-99b4-5a086e857473
    The video shows a man sitting at a table with a bowl of food in front of him. He is wearing a black t-shirt with a white text that reads "If it's not spicy, I'm not eating". The man has his hands raised in the air, making a gesture that suggests he is excited or enthusiastic about the food. The table is covered with a white tablecloth and there is a potted plant on the table. The background of the video shows a building with a thatched roof and a bicycle parked outside. The man appears to be in a casual outdoor setting, possibly a restaurant or a cafe. The overall style of the video is casual and informal, capturing a moment of enjoyment and anticipation.
    static
    3.4341225624084473 , BM25F_food_frequency:2, BM25F_food_propLength:75, BM25F_outside_frequency:1, BM25F_outside_propLength:75
    c93d4172-65a2-418f-b692-6832a33259b0
    The video shows a person preparing a sauce using a food processor. The food processor is placed on a green countertop. In the first frame, the person is pouring water into the food processor. In the second frame, the food processor is turned on and blending the ingredients. In the third frame, the person is pouring the blended sauce into a jar. The jar is placed on the countertop next to the food processor. The person is using a spoon to help pour the sauce into the jar. The entire process is captured in a close-up shot, focusing on the food processor and the jar. The style of the video is a simple, straightforward demonstration of a cooking technique.
    Undetermined
    2.3437130451202393 , BM25F_food_frequency:6, BM25F_food_propLength:48
    cf17c0f0-7770-44b9-ae90-4bb74a3f1a64
    The video shows a close-up of a bowl of food with a red spoon in it. The spoon is placed in the bowl, and the food appears to be a creamy, white substance with chunks of orange and yellow. The bowl is placed on a wooden table, and the background is blurred, focusing the viewer's attention on the bowl and spoon. The style of the video is simple and straightforward, with a focus on the food and the spoon, and no additional elements or actions are shown. The lighting is bright, and the colors are vibrant, making the food look appetizing. The video does not contain any text or additional elements, and the focus is solely on the bowl of food and the spoon.
    static
    2.2152414321899414 , BM25F_food_frequency:5, BM25F_food_propLength:59
    a5ba53fa-5f54-4b0f-abcb-d2eef8c3f377
    The video captures a lively street food scene. In the first frame, a man in a blue apron is seen preparing food at a small cart. He is surrounded by various ingredients and utensils, including a bowl of food and a spoon. In the second frame, a customer approaches the cart, holding a piece of paper, possibly a receipt or order slip. In the third frame, the customer is seen receiving the food from the man in the blue apron. The background of the video reveals a bustling street with other people and buildings, adding to the vibrant atmosphere of the scene. The style of the video is candid and dynamic, capturing the essence of street food culture.
    static
    2.1990089416503906 , BM25F_food_frequency:5, BM25F_food_propLength:62
    9d57d0ea-5127-4bcd-83d8-6509086f1ecc
    The video shows a close-up of a wok on a stove, filled with food being cooked. The food appears to be a mix of vegetables and possibly some meat, with the ingredients being cut into small pieces. The wok is on a burner, and the heat is causing the food to sizzle and cook. The style of the video is a simple, straightforward cooking tutorial, focusing on the food and the cooking process. The camera angle is from above, providing a clear view of the food and the wok. The lighting is bright, highlighting the colors of the food and the wok. The video does not contain any text or additional elements, focusing solely on the cooking process.
    tilt_up
    2.2746288776397705 , BM25F_food_frequency:6, BM25F_food_propLength:62
    5c33334e-f639-4e01-ab1f-90a18088a4ad
    The video is a close-up of a plate of food being served. The food consists of a variety of ingredients, including green beans, red peppers, and chunks of meat. The plate is placed on a blue table, and there is a small bowl of sauce next to it. The style of the video is simple and straightforward, focusing on the food and the table setting. The camera angle is slightly above the plate, providing a clear view of the food and the table. The lighting is bright, highlighting the colors of the food and the blue of the table. The video does not contain any text or additional elements. The focus is solely on the food and the presentation.
    static
    2.284247636795044 , BM25F_food_frequency:6, BM25F_food_propLength:60
    54df23cd-23c9-4d90-a109-2e6a1ea8f763
    The video shows a person cooking food on a stove. The person is using a spatula to flip the food, which appears to be a type of fried pastry or dough. The food is golden brown and has a crispy texture. The stove is made of metal and has a black surface. There is a red logo in the corner of the video that says "Street food." The style of the video is a close-up shot of the cooking process, focusing on the food and the person's hands. The lighting is bright and the colors are vibrant, highlighting the textures and colors of the food. The video captures the action of cooking and the transformation of the food from raw to cooked.
    pan_right
    2.3319075107574463 , BM25F_food_propLength:62, BM25F_food_frequency:7
    a2b05f99-993a-4011-9720-8bba57cbf79c
    The video is a cooking show featuring a man and three women. They are seated around a table with various food items and ingredients. The man is holding a plate of food and appears to be presenting it to the women. The women are smiling and seem to be enjoying the food. The table is covered with a white tablecloth and has a variety of items on it, including a bowl, a cup, and a bottle. The setting appears to be a kitchen or a dining area. The style of the video is casual and friendly, with the participants engaging in conversation and enjoying the food. The focus is on the food and the interaction between the participants.
    static
    2.226196765899658 , BM25F_food_frequency:5, BM25F_food_propLength:57
    62ff6c30-6074-498d-97a3-4bdfe8156dc9
    The video captures a close-up view of a plate of food being eaten. The plate contains a variety of dishes, including meat, vegetables, and rice. A spoon is being used to scoop up the food, and a fork is also visible on the plate. The food appears to be well-prepared and appetizing. The style of the video is realistic and straightforward, focusing on the details of the food and the act of eating. The colors and textures of the food are vividly displayed, making the video visually appealing. The video does not contain any text or additional elements, keeping the focus solely on the food and the eating process.
    Undetermined
    2.279428243637085 , BM25F_food_propLength:61, BM25F_food_frequency:6
    69aeca53-0998-4b4f-8dfc-79f0b1a86246
    The video captures a lively scene at a food court. In the first frame, a man in a red shirt is seen holding a plate of food, his eyes focused on the camera. In the second frame, he is seen taking a bite of his food, his expression one of enjoyment. In the third frame, he is seen holding his plate, ready to move on to the next food stall. The background is filled with other people, each engrossed in their own activities, adding to the bustling atmosphere of the food court. The video is shot in a realistic style, capturing the everyday life of the food court with its vibrant colors and dynamic movements.
    static
    2.2698497772216797 , BM25F_food_frequency:6, BM25F_food_propLength:63
    f14fe261-35bf-48c9-b20a-4ebb3c3c7b01
    In the video, a woman is seen standing outside a house, holding a tray of food. She is wearing a red sweater and a blue skirt. The house has a white door and a red flag hanging on the porch. There are several potted plants on the porch, and a Christmas tree is visible in the background. The woman appears to be in the process of delivering the food to the house. The video captures a moment of everyday life, with the woman's actions suggesting a sense of community and care. The setting is peaceful and homely, with the potted plants and Christmas tree adding a touch of warmth and festivity to the scene.
    pan_right
    3.715664863586426 , BM25F_food_frequency:2, BM25F_food_propLength:61, BM25F_outside_frequency:1, BM25F_outside_propLength:61
    358679b6-2fec-44d3-b120-d1f0c4216f1b
    The video captures a close-up view of a fork piercing through a piece of fried chicken. The chicken is golden brown and appears to be crispy on the outside. The fork is silver and has a few specks of food on it. The chicken is being lifted from a plate that has other pieces of fried chicken and potatoes on it. The plate is white with a blue rim. The background is blurred, but it seems to be a wooden table. The video is in color and has a shallow depth of field, focusing on the fork and the piece of chicken. The style of the video is realistic and it seems to be a food review or a cooking tutorial.
    static
    3.737610340118408 , BM25F_outside_propLength:60, BM25F_food_frequency:2, BM25F_food_propLength:60, BM25F_outside_frequency:1
    2710a7eb-0f72-413d-bcdc-4d560df3ef33
    The video features a man sitting on a couch inside a bus, which appears to be converted into a living space. He is wearing a white t-shirt and a black baseball cap. The man is gesturing with his hands, possibly explaining something or engaging in a conversation. The bus interior is well-lit, with natural light coming through the windows. The windows offer a view of the outside, where a desert landscape can be seen. The man's expression is one of surprise or excitement. The overall style of the video is casual and informal, with a focus on the man's reaction to something happening outside the bus.
    static
    2.6936416625976562 , BM25F_outside_frequency:2, BM25F_outside_propLength:68
    bc4f4e29-46e2-486b-8aab-2fb3c0d06b1b
    In the video, a woman is seen preparing food outdoors. She is wearing a red shirt and sunglasses, and she is holding a piece of food in her hand. The table in front of her is filled with various food items and condiments, including a bottle of oil, a bowl of fruit, and a tray of food. The setting appears to be a garden or park, with trees and plants in the background. The woman seems to be enjoying her time outdoors, engaging in a leisurely activity of preparing and tasting food. The overall style of the video is casual and relaxed, capturing a pleasant moment of outdoor cooking and dining.
    static
    2.204393148422241 , BM25F_food_frequency:5, BM25F_food_propLength:61
    fafbff89-81a3-4cb1-b9f5-0f8b5c834640
    The video shows a close-up of a plate of food being eaten. The plate contains a variety of dishes, including a red sauce, green olives, and a stuffed green pepper. A spoon is visible in the frame, indicating that the person is eating the food. The style of the video is a simple, straightforward food video, focusing on the presentation and consumption of the meal. The colors of the food are vibrant, and the close-up view allows for a detailed look at the textures and ingredients. The video does not contain any additional elements or background, keeping the focus solely on the food.
    static
    2.204393148422241 , BM25F_food_frequency:5, BM25F_food_propLength:61
    be23e943-35bd-4ac1-a9e1-d37e7e1a512a
    The video shows a person using chopsticks to pick up a portion of food from a bowl. The bowl is placed on a table, and the food appears to be a type of grain or rice. The person is holding the chopsticks with both hands, and the food is being lifted from the bowl. The style of the video is simple and straightforward, focusing on the action of using chopsticks to pick up food. The background is minimal, with the focus being on the person and the bowl of food. The video does not contain any additional elements or distractions, allowing the viewer to focus on the main action.
    static
    2.265409469604492 , BM25F_food_frequency:5, BM25F_food_propLength:50
    62380b7e-86ce-4c32-aa96-bf00df3b8faa
    The video shows a close-up of a plate of food on a wooden table. The food consists of noodles, vegetables, and a piece of meat. The noodles are yellow and appear to be cooked. The vegetables include green onions and a lime wedge. The meat is brown and looks like it could be chicken or beef. The plate is white and the food is arranged in a pile on top of the noodles. The table is wooden and the background is blurred, focusing the viewer's attention on the plate of food. The style of the video is a simple, straightforward food shot with no additional elements or actions.
    static
    2.22070574760437 , BM25F_food_frequency:5, BM25F_food_propLength:58
    e7d21f95-1390-4d78-9864-f921062a464f
    The video features a man with long hair and a mustache, who appears to be in a car. The man is looking to his left with a concerned expression on his face. The car's interior is visible, with the number 13 visible on the side. The man is wearing a black jacket and has a beard. The car is parked outside a building with a blue door. The man's expression suggests that he is in the middle of a conversation or reacting to something happening outside the car. The overall style of the video is realistic, with a focus on the man's facial expression and the interior of the car.
    static
    2.8638253211975098 , BM25F_outside_frequency:2, BM25F_outside_propLength:55
    ad920723-1bab-4f80-9f6b-a7abf0c3516c
    The video shows the interior of a car from the perspective of the back seat. The car has two seats, both of which are black and appear to be leather. The seats are designed with a modern and sleek style, featuring a headrest and a lumbar support. The car's interior is well-lit, with a blue light illuminating the ceiling. The windows of the car are tinted, providing a view of the outside world. The car is moving, as indicated by the blurred scenery outside the windows. The overall style of the video is sleek and modern, with a focus on the car's interior design.
    tilt_down+pan_left
    2.8499743938446045 , BM25F_outside_propLength:56, BM25F_outside_frequency:2
    cd2b1b4d-0a8c-4df3-9a65-fcff20645023
    In the video, a man and a woman are standing outside a building with a satellite dish on top. The man is holding a tortoise in his hands, showing it to the woman. The woman is looking at the tortoise with interest. The tortoise is brown and has a patterned shell. The man is wearing a green shirt and the woman is wearing a white shirt. The building they are standing in front of has a black roof. The satellite dish on top of the building is white. The video captures a moment of interaction between the man and the woman, with the tortoise as the focal point.
    static
    2.228468656539917 , BM25F_outside_frequency:1, BM25F_outside_propLength:47
    e046846d-00b6-46a3-9c1b-cb53ec87f932
    The video shows a close-up of a plate of food, which includes a serving of rice, beans, and avocado slices. The food is presented on a green plate with a striped pattern. The style of the video is simple and straightforward, focusing on the food without any additional context or background. The camera angle is slightly elevated, providing a clear view of the food on the plate. The lighting is bright, highlighting the colors and textures of the food. The video does not contain any text or additional elements, and the focus is solely on the plate of food.
    Undetermined
    2.31856369972229 , BM25F_food_frequency:6, BM25F_food_propLength:53
    622302a4-1a1d-43a2-9e18-7208c2343d18
    The video shows a person in a blue apron serving food from a buffet. The food consists of various fried items, including what appears to be fried potatoes and possibly other fried foods. The person is using a white plate to serve the food, and the food is displayed in two large metal trays. The setting suggests a casual dining environment, possibly a cafeteria or a buffet-style restaurant. The style of the video is straightforward and documentary, capturing the action of the person serving food without any additional embellishments or artistic effects.
    static
    2.231715440750122 , BM25F_food_frequency:5, BM25F_food_propLength:56
    96e7ab1a-05da-4bd7-ad97-26d3c36c2c02
    The video shows a woman in a kitchen, preparing food using a food processor. She is wearing a black apron and is focused on her task. The kitchen is well-equipped with wooden cabinets, a stainless steel refrigerator, and a sink. The woman is using a spoon to mix the ingredients in the food processor. The video is likely a tutorial or a cooking show, as it demonstrates the use of a food processor in a home kitchen setting. The style of the video is informative and practical, aimed at teaching viewers how to use kitchen appliances and prepare food.
    static
    2.231715440750122 , BM25F_food_frequency:5, BM25F_food_propLength:56
    fc50e30e-5312-43ff-af3c-de78b69c8c2a
    The video shows a cooking process, specifically frying food in a deep fryer. The first frame shows the fryer with a basket inside, filled with hot oil. The second frame shows the food being added to the fryer, with the basket submerged in the oil. The third frame shows the food being removed from the fryer, with the basket lifted out of the oil. The style of the video is a simple, straightforward demonstration of a cooking technique, with no additional context or embellishments. The focus is solely on the process of frying food in a deep fryer.
    static
    2.2034966945648193 , BM25F_food_frequency:4, BM25F_food_propLength:45
    843af525-6f16-4002-aa66-2c5bc843e6fd
    In the video, a young woman wearing glasses and a tie-dye shirt is seen enjoying a cup of coffee. She is seated at a window, which offers a view of the street outside. The street is lined with parked cars, and the woman seems to be taking a break from her day to savor her drink. The video captures a moment of relaxation and enjoyment, as the woman takes a sip of her coffee and gazes out at the world outside her window. The overall style of the video is casual and relaxed, with a focus on the woman's enjoyment of her coffee and the view outside.
    tilt_up+pan_right
    3.222257375717163 , BM25F_outside_frequency:3, BM25F_outside_propLength:58
    a03ef02a-5684-49cd-9524-f71dc2b178e6
    The video captures a close-up view of a spoonful of food, which appears to be a type of curry or stew. The food is yellow and green, with visible chunks of vegetables and grains. The spoon is silver and shiny, reflecting the light. The background is blurred, but it seems to be an outdoor setting with green foliage. The style of the video is a close-up food shot, focusing on the textures and colors of the food. The lighting is bright, highlighting the details of the food and the spoon. The video does not contain any text or additional elements.
    static
    2.22070574760437 , BM25F_food_frequency:5, BM25F_food_propLength:58
    c215ba0c-ace8-4b8d-850f-9c3121c6ba1c
    The video features a woman with blonde hair and blue eyes, who is wearing a black top. She is captured in a close-up shot, with her face filling the frame. The woman appears to be speaking, as suggested by her open mouth and the slight movement of her lips. The background of the video is blurred, but it seems to be an indoor setting with a window that offers a view of a snowy landscape outside. The overall style of the video is casual and intimate, focusing on the woman's expression and the contrast between her and the wintry scene outside.
    static
    2.7184951305389404 , BM25F_outside_frequency:2, BM25F_outside_propLength:66
    728b84f2-2df4-4d69-a87b-a97df075bb39
    The video shows a young man enjoying a meal at a restaurant. He is wearing a black t-shirt with the words "Eat More Tom" printed on it. The man is seated at a table, which is covered with a white tablecloth. In front of him is a plate of food, which includes a salad and a piece of meat. He is using a fork to eat the food. The restaurant has a large window, through which you can see the outside. The man appears to be enjoying his meal, as he is making a funny face while eating. The overall atmosphere of the video is casual and relaxed.
    static
    3.6304783821105957 , BM25F_food_frequency:2, BM25F_food_propLength:65, BM25F_outside_frequency:1, BM25F_outside_propLength:65
    a4db4edb-ffcf-4355-b199-6be5b349fd39
    The video shows a young woman standing in front of a building, smiling at the camera. She is wearing a black t-shirt with a graphic design on it and a baseball cap. The building behind her appears to be a fast food restaurant, with a red and white color scheme. There is a bench and a trash can visible in the background. The woman seems to be enjoying her time outside, possibly taking a break from work or just enjoying the day. The overall style of the video is casual and friendly, with a focus on the woman and her surroundings.
    static
    3.181295156478882 , BM25F_food_frequency:1, BM25F_food_propLength:64, BM25F_outside_frequency:1, BM25F_outside_propLength:64
    c1a17b9e-8e55-45f0-85ca-7a58d0e42d27
    The video shows a bento box with a variety of food items neatly arranged in compartments. The bento box is open, revealing its contents. The food items include cheese, boiled eggs, a salad, strawberries, carrots, and cookies. The bento box is placed on a table, and the background is blurred, focusing attention on the food. The style of the video is simple and straightforward, with no additional elements or actions. The focus is solely on the bento box and its contents, showcasing the variety and presentation of the food.
    zoom_out
    2.1834564208984375 , BM25F_food_frequency:4, BM25F_food_propLength:48
    87cdd187-8a27-430d-b7dd-3e755cda0ce1
    The video shows a young man sitting in the back seat of a car. He is wearing a blue baseball cap and a black jacket. The car is moving, and the man is looking out the window. The man appears to be deep in thought or lost in his own world. The car is on a road, and the scenery outside the window is blurry, indicating that the car is moving at a high speed. The man's expression is serious, and he seems to be focused on something outside the car. The video captures a moment of quiet introspection in the midst of a journey.
    pan_right
    2.836256980895996 , BM25F_outside_frequency:2, BM25F_outside_propLength:57
    47be8182-a8a7-4f64-83cc-2fac86f2bcf5
    The video features a woman standing in front of a restaurant with a vibrant mural on the wall. She is wearing a black t-shirt with a graphic design and is holding a black purse. Her expression is neutral, and she is looking directly at the camera. The restaurant has a red door and a window with a colorful display of food items. There are several tables and chairs outside the restaurant, suggesting it is a popular dining spot. The overall style of the video is casual and candid, capturing a moment in the woman's day.
    tilt_up+zoom_in
    3.2689261436462402 , BM25F_food_frequency:1, BM25F_food_propLength:60, BM25F_outside_frequency:1, BM25F_outside_propLength:60
    b868dfbb-cac9-479d-beef-2dac47413a99
    The video is a montage of a city street scene. In the first frame, a group of people in white coats are standing outside a car wash. They are dressed in white coats and are looking at the camera. In the second frame, a white car is parked in front of the car wash. The car is shiny and clean. In the third frame, the car is driving down the street. The street is lined with trees and buildings. The car is moving at a slow pace. The video is shot in a realistic style, capturing the everyday life of the city.
    Undetermined
    2.211740016937256 , BM25F_outside_frequency:1, BM25F_outside_propLength:48
    5699267e-8873-42ec-9aa2-0dfee54fe7cb
    In the video, a man is seen preparing food in a kitchen. He is wearing a blue and white striped shirt and glasses. The kitchen is equipped with a sink, a window, and a television mounted on the wall. The man is using a cutting board and a knife to chop up some food. The window provides a view of the outside, and the television is turned off. The man is focused on his task, and the kitchen appears to be well-lit. The overall style of the video is casual and everyday, capturing a common domestic scene.
    static
    3.851483106613159 , BM25F_food_frequency:2, BM25F_food_propLength:55, BM25F_outside_frequency:1, BM25F_outside_propLength:55
    bb6423c6-10b1-4c0c-9e94-cc04dcdea82b
    The video shows a person holding a black, round, flatbread-like food item with a variety of toppings, including cheese and vegetables. The food is held up against a blurred background, which suggests an outdoor setting. The style of the video is casual and informal, with a focus on the food item. The lighting is natural, and the overall mood of the video is relaxed and inviting. The video likely aims to showcase the food item and its toppings, possibly for a food review or a recipe demonstration.
    static
    2.248436212539673 , BM25F_food_propLength:53, BM25F_food_frequency:5
    afe1b240-6e4b-4235-a399-2b9311991b4d
    In the video, a woman is seen enjoying a meal at a restaurant with a window view. She is seated at a table, holding a fork with a piece of food on it. She is smiling, indicating that she is having a pleasant time. The restaurant has a potted plant hanging on the wall, adding a touch of greenery to the ambiance. The window offers a view of the outside, suggesting that the restaurant is located in a scenic area. The woman's attire and the setting suggest a casual and relaxed dining experience.
    static
    3.385496139526367 , BM25F_food_frequency:1, BM25F_food_propLength:55, BM25F_outside_frequency:1, BM25F_outside_propLength:55
    8ad751cb-388b-40dc-846e-feb4d47fbe30
    The video captures a close-up view of a plate of food being eaten with a fork and spoon. The food consists of a colorful mix of ingredients, including greens, red peppers, and white onions. The fork and spoon are used to pick up the food, and the person eating the food is not visible in the video. The style of the video is a simple, straightforward food-focused video, with no additional context or background provided. The focus is solely on the food and the utensils being used to eat it.
    Undetermined
    2.323550224304199 , BM25F_food_frequency:6, BM25F_food_propLength:52
    28d9957c-481a-4fe6-9fe7-117732c6fd2f
    In the video, a man is seen in a kitchen, holding a plate with a piece of meat on it. He is wearing a blue shirt and appears to be in the process of cooking or preparing the food. The kitchen is well-equipped with various appliances and utensils, including a refrigerator, an oven, and a sink. The man is standing in front of a window, which lets in natural light and provides a view of the outside. The overall style of the video is casual and homey, capturing a moment of everyday life.
    static
    3.3378844261169434 , BM25F_food_frequency:1, BM25F_food_propLength:57, BM25F_outside_frequency:1, BM25F_outside_propLength:57
    439618c3-4ffb-4220-8bf5-e490de6d214e
    In the video, two men are sitting at a table outside a house. The man on the left is wearing a black shirt and glasses, while the man on the right is wearing a white shirt and a red and white baseball cap. The man on the right is holding a knife and a piece of bread, and he is pointing at the bread with the knife. There is a bag of bread on the table in front of them. The house behind them is blue and white. The video captures a casual and friendly interaction between the two men.
    static
    2.2454521656036377 , BM25F_outside_frequency:1, BM25F_outside_propLength:46
    94ec8542-420e-4899-b3a8-ab74fb6b7047
    The video captures a street food scene where a person is preparing and serving food. The food consists of small, round, yellow items that are likely some type of fried or cooked snack. The person is using a large metal tray to hold the food, and they are wearing a black jacket and red pants. The setting appears to be outdoors, possibly on a street or in a market area. The style of the video is casual and documentary-like, capturing the everyday life of street food vendors.
    pan_left
    2.226196765899658 , BM25F_food_frequency:5, BM25F_food_propLength:57
    8b9e22b0-4675-4f84-9f29-e044b5af99fa
    The video shows a man with a beard and mustache, wearing a black shirt. He is in a kitchen with white cabinets. In the first frame, he is looking down, possibly at a plate of food. In the second frame, he is looking up, with his eyes closed, as if he is savoring the taste of the food. In the third frame, he is pointing at something, possibly the food or a utensil. The style of the video is casual and candid, capturing a moment of enjoyment or appreciation of the food.
    static
    2.1834564208984375 , BM25F_food_frequency:4, BM25F_food_propLength:48
    8f48809a-f87b-4083-b2cf-2fd54312a328
    In the video, a man and a woman are seen sitting in the backseat of a car. The man is wearing a white hoodie and a baseball cap, while the woman is wearing a black and white sweater and sunglasses. They are both looking out of the car window, seemingly engaged in a conversation. The car is moving along a road, and the scenery outside the window includes trees and buildings. The overall style of the video is casual and candid, capturing a moment of everyday life.
    static
    2.228468656539917 , BM25F_outside_frequency:1, BM25F_outside_propLength:47
    37619cc1-1319-48e5-ba93-3352fce83de3
    The video shows a man sitting in the back seat of a car, wearing a blue shirt and a seatbelt. The car has a sunroof open, and the interior is visible. The man appears to be speaking or gesturing with his hands. The style of the video is a casual, personal vlog or travel video, capturing the man's experience from the back seat of the car. The focus is on the man and his interaction with the car's interior, rather than the surroundings outside the car.
    static
    2.228468656539917 , BM25F_outside_frequency:1, BM25F_outside_propLength:47
    cc51f845-d6ee-4b65-8587-87168baf2e5c
    In the video, a man is seen inside a vehicle, holding a yellow device in his hand. He appears to be engaged in a conversation with another man who is standing outside the vehicle. The vehicle is parked in a grassy area, and the man inside is gesturing with his hand, possibly explaining something to the man outside. The man outside is attentively listening to the man inside. The scene suggests a casual, friendly interaction between the two men.
    static
    3.408651113510132 , BM25F_outside_propLength:43, BM25F_outside_frequency:3
    fdc6df1d-9f9b-4e25-8aa4-96590539a0cb
    The video shows a young man sitting on a train, smiling and pointing at the camera with his index finger. He is wearing a black hoodie and has short blonde hair. The train interior is visible, with purple seats and a window showing the outside. The man appears to be in a good mood, possibly engaging with someone outside the frame. The style of the video is casual and candid, capturing a moment of interaction between the man and the viewer.
    static
    2.891934871673584 , BM25F_outside_frequency:2, BM25F_outside_propLength:53
    52ff6558-d2a5-4b6f-92cd-a1c54c97fb48
    In the video, a woman is seen driving a car with her two children in the backseat. The woman is wearing sunglasses and a gray tank top, while the children are dressed in casual clothing. The car is moving along a road, and the woman is seen pointing at something outside the car, drawing the attention of her children. The video captures a candid moment of a family outing, with the woman and her children enjoying their journey together.
    static
    2.280207872390747 , BM25F_outside_propLength:44, BM25F_outside_frequency:1
    80b20811-b3e2-4859-9892-7c753c2e54d5
    In the video, two men are standing outside a house, engaged in a conversation. The man on the left is pointing at something, possibly a door or a window, while the man on the right is holding a broom. The house has a brick exterior and a white door. The man on the right is wearing a blue t-shirt with a robot design on it. The overall style of the video is casual and informal, capturing a moment of interaction between the two men.
    static
    2.195260763168335 , BM25F_outside_propLength:49, BM25F_outside_frequency:1
    b793a133-819d-4eab-9a5d-52db57acea78
    The video depicts a hospital room with a large window and a view of a building outside. The room is furnished with a hospital bed, a chair, and a table. The bed is covered with a white sheet and a yellow pillow. The chair is green and has a metal frame. The table is white and has a metal frame. The room has a blue and white curtain. The window is open, allowing natural light to enter the room. The room appears to be empty.
    static
    2.3719940185546875 , BM25F_outside_frequency:1, BM25F_outside_propLength:39
    b2a232ba-c99b-43e6-9dde-72fb9bda7211
    The video shows a man with a beard and a bun, shirtless, sitting in a boat cabin. He is holding a walkie-talkie to his ear, engaged in a conversation. The cabin is dimly lit, with a window showing the outside. The man appears to be focused on the conversation, and the walkie-talkie is clearly visible in his hand. The overall style of the video is candid and informal, capturing a moment of communication in a serene setting.
    static
    2.2454521656036377 , BM25F_outside_frequency:1, BM25F_outside_propLength:46
    f622b1ec-718f-4908-9820-1f191383d058
    The video features a man in a black shirt standing in front of a large window. The man is speaking and appears to be in the middle of a conversation. The window behind him offers a view of the outside, which is bright and sunny. The man's expression is serious, and he seems to be focused on his conversation. The overall style of the video is straightforward and professional, with a focus on the man and his speech.
    static
    2.280207872390747 , BM25F_outside_propLength:44, BM25F_outside_frequency:1
    d95b6a46-c3a1-4d5b-8ff0-007f924f8137
    In the video, a man with blonde hair and a beard is seen driving a vehicle. He is wearing a blue vest and is focused on the road ahead. The interior of the vehicle is visible, with a gray seat and a window that offers a glimpse of the outside world. The man's expression is serious, suggesting that he is concentrating on his driving. The overall style of the video is realistic, capturing a moment of everyday life.
    static
    2.2454521656036377 , BM25F_outside_frequency:1, BM25F_outside_propLength:46
    8946320a-367c-4bbc-901d-dd6a1946224b
    In the video, a man and a woman are sitting in white plastic chairs on a sidewalk. The man is wearing a gray t-shirt and shorts, while the woman is wearing a blue long-sleeved shirt. Both of them have green stickers on their shirts. The man is wearing a watch on his left wrist. In the background, there is a black car parked on the street. The video captures a casual, everyday scene of two people sitting outside.
    static
    2.195260763168335 , BM25F_outside_frequency:1, BM25F_outside_propLength:49
    4718f6c9-0d5c-4897-b57a-2e98106948eb
    The video shows a man with a beard sitting in the back seat of a car. He is wearing a white t-shirt and has his arm resting on the back of the seat. The car appears to be in motion, as suggested by the blurred background. The man seems to be engaged in a conversation or perhaps reacting to something happening outside the car. The style of the video is candid and informal, capturing a moment of everyday life.
    static
    2.179025173187256 , BM25F_outside_frequency:1, BM25F_outside_propLength:50
    9d7323fe-cf9c-4edd-8ac0-e4c15fc0497a
    In the video, a woman with blonde hair is seen in a kitchen setting. She is wearing a green and blue patterned top and is holding a fork in her hand. The kitchen has a window that offers a view of trees outside. The woman appears to be in the middle of a cooking or eating process, as she is holding the fork. The overall style of the video is casual and homey, capturing a moment in the woman's daily life.
    static
    2.195260763168335 , BM25F_outside_propLength:49, BM25F_outside_frequency:1
    75afecbe-2645-4bbc-be48-3ac8510a6e41
    The video shows two men sitting outside, with one man blowing smoke from his mouth. The man blowing smoke is wearing a black t-shirt and a white baseball cap. The other man is wearing a gray t-shirt with a white graphic on it. They are both sitting on a bench, and the background features a green wooden fence. The style of the video is casual and candid, capturing a moment of leisure between the two men.
    static
    2.228468656539917 , BM25F_outside_frequency:1, BM25F_outside_propLength:47
    daa3c0b5-964b-4161-93fc-6a2f15d9054f
    In the video, a group of four friends, two men and two women, are standing outside a building with a green gate. They are all dressed casually, with the men wearing glasses and the women wearing sunglasses. The friends are engaged in a conversation, with one of the men gesturing towards the gate. The building behind them has a window, and the overall atmosphere of the video is relaxed and friendly.
    static
    2.3530502319335938 , BM25F_outside_frequency:1, BM25F_outside_propLength:40
    f12deea2-6a35-493d-9eb9-cf77d3617c1e
    The video shows a woman wearing large purple sunglasses, sitting in a vehicle. She is wearing a green shirt and appears to be looking directly at the camera. The vehicle has a window behind her, which shows a view of the outside. The interior of the vehicle is visible, with a seatbelt strap across her chest. The style of the video is casual and candid, capturing a moment in the woman's day.
    static
    2.2454521656036377 , BM25F_outside_frequency:1, BM25F_outside_propLength:46
    cde0a1d3-0643-4f64-9ae8-9c8467b2b6d8
    The video shows a close-up of a plate of food being eaten. The plate contains scrambled eggs, fried potatoes, and a side of grits. The food is being eaten with a fork. The person eating the food is not visible in the video. The style of the video is a simple, straightforward food video with no additional elements or distractions. The focus is solely on the food and the act of eating it.
    static
    2.323885440826416 , BM25F_food_frequency:5, BM25F_food_propLength:40
    286fc0cc-d2a9-4df7-bb2c-be2e03e2a6c1
    The video features a man in a suit, sitting in a room with a window in the background. The man is looking to his left, and his expression is serious. The room has a brick wall, and there is a tree outside the window. The man is wearing a white shirt and a dark suit. The video is shot in a realistic style, with natural lighting and a focus on the man's face and expression.
    static
    2.3719940185546875 , BM25F_outside_frequency:1, BM25F_outside_propLength:39
    ffe739f3-96c1-4bc8-9a43-f73b6ba15c3a
    The video shows a man in a gray t-shirt with a beard, sitting at a table outside a building. He is holding a spoon to his mouth, about to take a bite of his food. The table is set with a white tablecloth and there are yellow flowers in a pot nearby. The building has a green awning and a brick wall. The man appears to be enjoying his meal in a relaxed and casual setting.
    static
    3.5106871128082275 , BM25F_food_frequency:1, BM25F_food_propLength:50, BM25F_outside_frequency:1, BM25F_outside_propLength:50
    e0dc45f2-0206-4706-90df-39705c758254
    The video shows a man driving a car on a road. He is wearing a gray shirt and a black seatbelt. The car has a black steering wheel and a black dashboard. The man is looking out of the window, which is open. Outside, there is a person walking on the sidewalk. The road is lined with grass and trees. The sky is clear and blue. The man is driving the car at a steady pace.
    static
    2.316056489944458 , BM25F_outside_propLength:42, BM25F_outside_frequency:1
    c4ecca72-57b6-4be8-9dc6-7ad2b1c18007
    The video shows a woman sitting in a kitchen, looking out the window. She is wearing a black shirt and a gray sweater. The kitchen has a black countertop and a white stove. The window has a striped curtain. The woman is smiling and appears to be enjoying the view outside. The video is shot in a realistic style, capturing the everyday life of the woman in her kitchen.
    static
    2.3530502319335938 , BM25F_outside_frequency:1, BM25F_outside_propLength:40
    b2f5450b-0302-4ba9-b69b-1bbd901ab7b3
    In the video, a man is seen wearing a black puffy jacket with a hood, standing outdoors in a grassy area with trees in the background. He is holding a tablet in his hands, which he appears to be using. The man is smiling and seems to be enjoying his time outside. The overall style of the video is casual and relaxed, capturing a moment of leisure in a natural setting.
    pan_left
    2.2626967430114746 , BM25F_outside_frequency:1, BM25F_outside_propLength:45
    f036d5fd-361c-4eb4-8531-89be09751f8b
    The video shows a man with a beard and mustache, wearing a gray shirt, in a room with a wooden wall. He is making a surprised or shocked facial expression, with his mouth open and eyes wide. The room has a red door and a window with a view of a person outside. The style of the video is candid and informal, capturing a spontaneous moment of the man's reaction.
    pan_right
    2.2626967430114746 , BM25F_outside_frequency:1, BM25F_outside_propLength:45
    c477618d-8deb-4692-b70b-51740a8db283
    The video shows a woman in a car, who appears to be crying. She is holding her hand up to her face, covering her eyes. The car's window is open, and the outside view shows a cloudy sky. The woman is wearing a blue shirt and has a ring on her finger. The style of the video is candid and personal, capturing a moment of emotion in a real-life setting.
    static
    2.2454521656036377 , BM25F_outside_frequency:1, BM25F_outside_propLength:46
    2abc3493-23cb-47fa-ba6f-0fe0933946d2
    In the video, a man in a striped shirt stands in a room with a yellow wall and a clock on the wall. He appears to be in a state of shock or surprise, as he looks off to the side with his mouth open. The room has a door that leads outside, where a tree can be seen. The man's expression and the setting suggest that something unexpected has occurred.
    tilt_up+pan_right
    2.228468656539917 , BM25F_outside_frequency:1, BM25F_outside_propLength:47
    d8ade316-c9a2-45b3-b046-833b6ef30151
    In the video, a man with a beard and a surprised expression is seen in a car. He is wearing a blue shirt and is looking to his left. The car's interior is visible, with a green door and a black seat. The man's expression suggests that he is reacting to something happening outside the car. The video captures a moment of surprise and curiosity.
    static
    2.3719940185546875 , BM25F_outside_frequency:1, BM25F_outside_propLength:39
    a01482a9-b036-430f-a3bd-d13c2d7232d8
    The video shows a young girl in a purple dress, standing on a sidewalk. She is holding a pink toy in her hand. The girl is wearing a pink bow in her hair. In the background, there are cars parked on the street. The girl is looking at the camera and appears to be speaking. The video is a simple, candid shot of a child playing outside.
    pan_left
    2.297992467880249 , BM25F_outside_frequency:1, BM25F_outside_propLength:43
    fdb744d5-2b80-4b32-b08e-05b173dfd31d
    The video shows a man standing outside on a cloudy day. He is wearing a tank top and has a beard. He is looking up at the sky with a smile on his face. The sky is filled with clouds, and the man appears to be enjoying the view. The overall style of the video is casual and relaxed, capturing a moment of happiness and contentment.
    static
    2.316056489944458 , BM25F_outside_frequency:1, BM25F_outside_propLength:42
    c4fedda5-4f2f-4587-935d-9e1a75282d13
    The video is a news segment featuring a man named Mike Stenhouse, who is the Executive Director of the RI Center for Freedom and Prosperity. The segment is about a Texas church shooting. The man is standing outside a yellow house, wearing a suit and tie, and has a beard. The video is a news report with a serious tone.
    static
    2.391245126724243 , BM25F_outside_frequency:1, BM25F_outside_propLength:38
    


```python
from weaviate.classes.query import MetadataQuery, Rerank

vector_names = ["Caption"]
response = jeopardy.query.bm25(
    query="food static",   
    limit=100,  
    query_properties=["caption"],  
    rerank=Rerank(
        prop="caption",
        query="food, room, outside"
    ),
    return_metadata=MetadataQuery(distance=True, explain_score=True,score=True),
)



for o in response.objects:
    print(o.uuid)
    print(o.properties['caption'])
    print(o.properties['cameraMotion'])
    print(o.metadata.score, o.metadata.explain_score)
```
