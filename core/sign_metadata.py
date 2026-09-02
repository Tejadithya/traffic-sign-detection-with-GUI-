"""
Metadata and categorization for German Traffic Sign Recognition Benchmark (GTSRB) classes.
Classes are indexed 0 to 42.
"""
from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class SignInfo:
    class_id: int
    name: str
    category: str
    action_instruction: str
    badge_color: str
    shape: str

SIGN_CLASSES: Dict[int, SignInfo] = {
    0: SignInfo(0, 'Speed limit (20km/h)', 'Speed Limit', 'Do not exceed 20 km/h. Maintain extreme caution in pedestrian zones.', '#E63946', 'Circular'),
    1: SignInfo(1, 'Speed limit (30km/h)', 'Speed Limit', 'Maximum speed 30 km/h. Watch for residential zones and cyclists.', '#E63946', 'Circular'),
    2: SignInfo(2, 'Speed limit (50km/h)', 'Speed Limit', 'Standard urban speed limit. Maintain maximum 50 km/h.', '#E63946', 'Circular'),
    3: SignInfo(3, 'Speed limit (60km/h)', 'Speed Limit', 'Maximum allowable speed is 60 km/h.', '#E63946', 'Circular'),
    4: SignInfo(4, 'Speed limit (70km/h)', 'Speed Limit', 'Maximum allowable speed is 70 km/h.', '#E63946', 'Circular'),
    5: SignInfo(5, 'Speed limit (80km/h)', 'Speed Limit', 'Maximum speed 80 km/h. Typical on secondary rural roads.', '#E63946', 'Circular'),
    6: SignInfo(6, 'End of speed limit (80km/h)', 'Derestriction', 'End of 80 km/h speed restriction. Resume standard road limits.', '#6C757D', 'Circular'),
    7: SignInfo(7, 'Speed limit (100km/h)', 'Speed Limit', 'Maximum speed 100 km/h on expressways / rural arterial roads.', '#E63946', 'Circular'),
    8: SignInfo(8, 'Speed limit (120km/h)', 'Speed Limit', 'Maximum speed 120 km/h on motorways.', '#E63946', 'Circular'),
    9: SignInfo(9, 'No passing', 'Prohibitory', 'Overtaking of moving vehicles is strictly forbidden for all motor vehicles.', '#E63946', 'Circular'),
    10: SignInfo(10, 'No passing veh over 3.5 tons', 'Prohibitory', 'Vehicles exceeding 3.5 metric tons may not overtake.', '#E63946', 'Circular'),
    11: SignInfo(11, 'Right-of-way at intersection', 'Priority', 'You have priority at the upcoming intersection only.', '#F4A261', 'Triangular'),
    12: SignInfo(12, 'Priority road', 'Priority', 'You have right-of-way on this road until cancelled by an end-of-priority sign.', '#E76F51', 'Diamond'),
    13: SignInfo(13, 'Yield', 'Priority', 'Give way to traffic on the main or intersecting roadway.', '#E63946', 'Inverted Triangle'),
    14: SignInfo(14, 'Stop', 'Priority', 'Come to a complete stop before the stop line. Yield to all conflicting traffic.', '#D00000', 'Octagonal'),
    15: SignInfo(15, 'No vehicles', 'Prohibitory', 'Road closed to all vehicles in both directions.', '#E63946', 'Circular'),
    16: SignInfo(16, 'Veh > 3.5 tons prohibited', 'Prohibitory', 'Heavy transport / commercial goods vehicles over 3.5t prohibited.', '#E63946', 'Circular'),
    17: SignInfo(17, 'No entry', 'Prohibitory', 'Entry strictly forbidden for all vehicles (one-way opposing traffic).', '#D00000', 'Circular'),
    18: SignInfo(18, 'General caution', 'Warning', 'Unspecified hazard ahead. Reduce speed and heighten road awareness.', '#F4A261', 'Triangular'),
    19: SignInfo(19, 'Dangerous curve left', 'Warning', 'Sharp curve to the left ahead. Decelerate before steering into bend.', '#F4A261', 'Triangular'),
    20: SignInfo(20, 'Dangerous curve right', 'Warning', 'Sharp curve to the right ahead. Decelerate before steering into bend.', '#F4A261', 'Triangular'),
    21: SignInfo(21, 'Double curve', 'Warning', 'Succession of dangerous bends ahead. Adjust speed and stay in lane.', '#F4A261', 'Triangular'),
    22: SignInfo(22, 'Bumpy road', 'Warning', 'Uneven road surface or speed bumps ahead. Reduce speed to avoid suspension damage.', '#F4A261', 'Triangular'),
    23: SignInfo(23, 'Slippery road', 'Warning', 'Surface may be slick due to moisture, oil, or weather. Avoid sudden maneuvers.', '#F4A261', 'Triangular'),
    24: SignInfo(24, 'Road narrows on the right', 'Warning', 'Roadway width decreases on the right side ahead. Merge smoothly.', '#F4A261', 'Triangular'),
    25: SignInfo(25, 'Road work', 'Warning', 'Construction or maintenance active on roadway. Watch for workers and machinery.', '#F4A261', 'Triangular'),
    26: SignInfo(26, 'Traffic signals', 'Warning', 'Traffic light control ahead. Prepare to stop if signal turns amber or red.', '#F4A261', 'Triangular'),
    27: SignInfo(27, 'Pedestrians', 'Warning', 'Pedestrian crossing or foot traffic likely. Yield to persons crossing.', '#F4A261', 'Triangular'),
    28: SignInfo(28, 'Children crossing', 'Warning', 'Near schools, playgrounds or residential areas. Expect sudden child crossings.', '#F4A261', 'Triangular'),
    29: SignInfo(29, 'Bicycles crossing', 'Warning', 'Cyclist crossing ahead. Check blind spots and yield appropriate clearance.', '#F4A261', 'Triangular'),
    30: SignInfo(30, 'Beware of ice/snow', 'Warning', 'Risk of black ice or slippery snow pack. Increase following distance.', '#F4A261', 'Triangular'),
    31: SignInfo(31, 'Wild animals crossing', 'Warning', 'Wildlife crossing area. Stay vigilant, especially during dawn, dusk, and night.', '#F4A261', 'Triangular'),
    32: SignInfo(32, 'End speed + passing limits', 'Derestriction', 'All previous speed and passing restrictions are lifted.', '#6C757D', 'Circular'),
    33: SignInfo(33, 'Turn right ahead', 'Mandatory', 'Mandatory movement: vehicles must turn right ahead.', '#1D3557', 'Circular'),
    34: SignInfo(34, 'Turn left ahead', 'Mandatory', 'Mandatory movement: vehicles must turn left ahead.', '#1D3557', 'Circular'),
    35: SignInfo(35, 'Ahead only', 'Mandatory', 'Mandatory movement: vehicles must proceed straight ahead only.', '#1D3557', 'Circular'),
    36: SignInfo(36, 'Go straight or right', 'Mandatory', 'Mandatory movement: proceed straight or turn right.', '#1D3557', 'Circular'),
    37: SignInfo(37, 'Go straight or left', 'Mandatory', 'Mandatory movement: proceed straight or turn left.', '#1D3557', 'Circular'),
    38: SignInfo(38, 'Keep right', 'Mandatory', 'Keep right of the traffic island, bollard, or obstacle.', '#1D3557', 'Circular'),
    39: SignInfo(39, 'Keep left', 'Mandatory', 'Keep left of the traffic island, bollard, or obstacle.', '#1D3557', 'Circular'),
    40: SignInfo(40, 'Roundabout mandatory', 'Mandatory', 'Roundabout ahead. Yield to circulatory traffic and obey direction of flow.', '#1D3557', 'Circular'),
    41: SignInfo(41, 'End of no passing', 'Derestriction', 'Prohibition on overtaking other vehicles is lifted.', '#6C757D', 'Circular'),
    42: SignInfo(42, 'End no passing veh > 3.5 tons', 'Derestriction', 'Heavy goods vehicles overtaking restriction is lifted.', '#6C757D', 'Circular')
}

def get_sign_info(class_id: int) -> SignInfo:
    return SIGN_CLASSES.get(class_id, SignInfo(
        class_id=class_id,
        name=f"Unknown Sign ({class_id})",
        category="Unknown",
        action_instruction="No traffic rule information available for this class.",
        badge_color="#6C757D",
        shape="Unknown"
    ))
