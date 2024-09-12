from dataclasses import dataclass
from langchain_community.document_loaders import GutenbergLoader
@dataclass
class TextTextNovel:

    # Project Gutenberg Wizard of Oz book
    example_text_loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/55/pg55.txt")
    example_text = example_text_loader.load()
    testing_text: str = example_text[0].page_content

@dataclass
class TestText:
    testing_text: str = """
    Trees play a crucial role in maintaining ecological balance and supporting life on Earth. 
    Their importance extends beyond mere aesthetics, as they are integral to environmental health, economic stability, and social well-being. 
    Protecting trees is essential for several reasons, each of which underscores their multifaceted contributions to life on our planet.
    
    Environmental Benefits
    
    Trees are vital for the environment as they produce oxygen, absorb carbon dioxide, and help mitigate climate change. 
    Through photosynthesis, trees convert carbon dioxide into oxygen, a process that is essential for the survival of most living organisms. 
    Moreover, trees act as carbon sinks, storing carbon that would otherwise contribute to global warming. 
    By reducing the concentration of greenhouse gases in the atmosphere, trees help stabilize the climate, making them a natural ally in the fight against climate change.
    Additionally, trees play a significant role in maintaining biodiversity. 
    They provide habitat and food for a wide range of species, from insects to birds and mammals. 
    Forests, which are large expanses of trees, are home to the majority of the world's terrestrial biodiversity. 
    Protecting trees, therefore, means preserving the habitats of countless species, some of which are endangered.
    
    Economic Importance
    
    Trees also have substantial economic value. 
    They contribute to the economy by providing raw materials for various industries, including timber, paper, and pharmaceuticals. 
    Sustainable forestry practices ensure that these resources are available for future generations without depleting the forests. 
    Furthermore, trees enhance property values and attract tourism, which can be a significant source of income for many communities.
    Urban areas benefit economically from trees as well. 
    They reduce the need for air conditioning by providing shade and cooling the environment, which can lead to significant energy savings. 
    Trees also help manage stormwater, reducing the costs associated with water treatment and flood damage.
    
    Social and Health Benefits
    
    On a social level, trees contribute to the well-being of communities. 
    They enhance the beauty of landscapes and provide spaces for recreation and relaxation. 
    Studies have shown that green spaces with abundant trees can reduce stress, improve mood, and promote physical activity, all of which contribute to better mental and physical health.
    Trees also play a role in social cohesion. 
    Community tree-planting projects can bring people together, fostering a sense of community and shared purpose. 
    These activities not only improve the local environment but also strengthen social bonds and promote environmental stewardship.
    
    Conclusion
    
    In conclusion, the protection of trees is of paramount importance due to their extensive environmental, economic, and social benefits. 
    As vital components of the Earth's ecosystem, trees help combat climate change, support biodiversity, and contribute to human health and well-being. 
    Efforts to protect and preserve trees, whether through conservation, sustainable forestry, or urban planning, are essential for ensuring a sustainable future for all living beings. 
    By recognizing and acting upon the importance of trees, we can create a healthier, more balanced world.
    The history of Namibia is a complex tapestry woven from the threads of indigenous heritage, colonial conquest, and the struggle for independence. 
    Situated on the southwestern coast of Africa, Namibia's history is marked by significant events and transitions that have shaped its path to becoming a sovereign nation.
    
    Pre-Colonial Era
    
    The earliest inhabitants of Namibia were the San and Khoe peoples, who lived as nomadic hunters and gatherers for thousands of years. 
    These indigenous groups managed the land's limited resources and developed a rich cultural heritage. 
    Over time, Bantu-speaking groups such as the Owambo, Herero, and Kavango migrated into the region, establishing agricultural and pastoral societies. 
    The interactions among these groups laid the foundation for Namibia's diverse cultural landscape.
    
    German Colonial Rule
    
    In the late 19th century, Namibia became a German colony known as German South West Africa. 
    The German colonial administration was marked by harsh policies and brutal suppression of uprisings, most notably the Herero and Namaqua genocide from 1904 to 1908, where tens of thousands of Herero and Nama people were killed. 
    This dark chapter in Namibia's history left a lasting impact on its indigenous populations and remains a poignant reminder of the atrocities of colonialism.
    
    South African Administration
    
    Following Germany's defeat in World War I, the League of Nations mandated South Africa to administer the territory. 
    South Africa imposed its apartheid policies, further marginalizing the indigenous peoples and intensifying the struggle for independence. 
    Despite international condemnation, South African control persisted, leading to decades of resistance and conflict.
    
    The Struggle for Independence
    
    The fight for Namibian independence gained momentum in the mid-20th century with the formation of the South West Africa People's Organization (SWAPO) in 1960. 
    SWAPO led an armed struggle against South African rule, drawing international attention to Namibia's plight. 
    The United Nations played a crucial role in advocating for Namibia's independence, culminating in a UN-supervised election in 1989.
    
    Independence and Beyond
    
    On March 21, 1990, Namibia achieved independence, with Sam Nujoma becoming its first president. 
    The transition was marked by efforts to foster national reconciliation and build a democratic society. 
    Namibia adopted a constitution that emphasized human rights and sought to address the legacies of colonialism and apartheid.
    In the years following independence, Namibia has made strides in economic development and social progress, although challenges remain. 
    The government has focused on land reform, addressing inequalities, and promoting sustainable development. 
    Namibia's journey from colonial subjugation to independence is a testament to the resilience and determination of its people.
    
    Conclusion
    
    Namibia's history is a narrative of endurance and transformation. 
    From its indigenous roots to colonial oppression and eventual liberation, Namibia's past has shaped its identity and continues to influence its future. 
    As Namibia moves forward, it carries the lessons of its history, striving to build a nation that honors its diverse heritage and fosters unity and prosperity for all its citizens
    Trees play a pivotal role in the field of medicine, serving as a vital source of both traditional and modern pharmaceutical products. 
    Their protection is crucial not only for maintaining biodiversity and ecological balance but also for ensuring the continued availability of medicinal resources that are fundamental to human health and well-being.
    
    Historical Significance of Medicinal Trees
    
    Historically, trees have been integral to traditional medicine systems across the globe. 
    Ancient cultures, such as those in China, India, and Egypt, have long utilized tree-derived substances for healing purposes. 
    For instance, the bark of the willow tree, which contains salicylic acid, has been used for centuries to alleviate pain and inflammation, eventually leading to the development of aspirin, one of the most widely used drugs today. 
    Similarly, the opium poppy, another plant-based source, has given rise to powerful painkillers like morphine and codeine, highlighting the long-standing relationship between trees and medicinal advancements.
    
    Modern Pharmaceutical Applications
    
    In contemporary medicine, trees continue to be a cornerstone for drug discovery and development. 
    A significant portion of pharmaceuticals are derived from plant compounds, with trees providing essential chemical templates for synthesizing new drugs. 
    These natural products often serve as the basis for treatments of serious ailments, including cancer, heart disease, and malaria. 
    For example, the Pacific yew tree is the source of paclitaxel, a chemotherapy drug used to treat various cancers. 
    This underscores the necessity of preserving tree species that may harbor potential cures for diseases yet to be fully understood.
    
    Biodiversity and Ecosystem Health
    
    The preservation of trees is also critical for maintaining the biodiversity necessary for medicinal plant research. 
    Studies have shown that old-growth forests, with their dense canopy cover, are rich in medicinal plant diversity, providing habitats for numerous species that could hold the key to future medical breakthroughs. 
    Deforestation and habitat destruction threaten these ecosystems, potentially leading to the loss of invaluable plant species before their medicinal properties can be explored.
    
    Socioeconomic and Cultural Importance
    
    Beyond their direct medicinal applications, trees support traditional medicine practices that are vital to the healthcare systems of many indigenous and rural communities. 
    In regions where access to modern pharmaceuticals is limited, trees offer an accessible and affordable source of treatment, often forming the backbone of local healthcare. 
    Protecting these natural resources ensures that communities can continue to rely on traditional knowledge and practices that have been passed down through generations.
    """