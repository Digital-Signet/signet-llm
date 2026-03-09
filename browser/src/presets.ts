/**
 * Pre-loaded training text datasets.
 */

export const PRESETS: Record<string, string> = {
  shakespeare: `To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause—there's the respect
That makes calamity of so long life.
For who would bear the whips and scorns of time,
Th'oppressor's wrong, the proud man's contumely,
The pangs of dispriz'd love, the law's delay,
The insolence of office, and the spurns
That patient merit of th'unworthy takes,
When he himself might his quietus make
With a bare bodkin? Who would fardels bear,
To grunt and sweat under a weary life,
But that the dread of something after death,
The undiscovere'd country, from whose bourn
No traveller returns, puzzles the will,
And makes us rather bear those ills we have
Than fly to others that we know not of?
Thus conscience does make cowards of us all,
And thus the native hue of resolution
Is sicklied o'er with the pale cast of thought,
And enterprises of great pitch and moment
With this regard their currents turn awry
And lose the name of action.

O Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name.
Or if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.
What's in a name? That which we call a rose
By any other name would smell as sweet.
So Romeo would, were he not Romeo call'd,
Retain that dear perfection which he owes
Without that title. Romeo, doff thy name;
And for that name, which is no part of thee,
Take all myself.

All the world's a stage,
And all the men and women merely players;
They have their exits and their entrances;
And one man in his time plays many parts,
His acts being seven ages. At first the infant,
Mewling and puking in the nurse's arms;
Then the whining school-boy with his satchel
And shining morning face, creeping like snail
Unwillingly to school. And then the lover,
Sighing like furnace, with a woeful ballad
Made to his mistress' eyebrow. Then a soldier,
Full of strange oaths, and bearded like the pard,
Jealous in honour, sudden and quick in quarrel,
Seeking the bubble reputation
Even in the cannon's mouth. And then the justice,
In fair round belly with good capon lin'd,
With eyes severe and beard of formal cut,
Full of wise saws and modern instances;
And so he plays his part.

Is this a dagger which I see before me,
The handle toward my hand? Come, let me clutch thee.
I have thee not, and yet I see thee still.
Art thou not, fatal vision, sensible
To feeling as to sight? or art thou but
A dagger of the mind, a false creation,
Proceeding from the heat-oppressed brain?

If music be the food of love, play on,
Give me excess of it; that surfeiting,
The appetite may sicken, and so die.`,

  rhymes: `Twinkle twinkle little star how I wonder what you are
Up above the world so high like a diamond in the sky
Twinkle twinkle little star how I wonder what you are

Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
All the king's horses and all the king's men
Couldn't put Humpty together again

Jack and Jill went up the hill
To fetch a pail of water
Jack fell down and broke his crown
And Jill came tumbling after

Mary had a little lamb its fleece was white as snow
And everywhere that Mary went the lamb was sure to go
It followed her to school one day which was against the rules
It made the children laugh and play to see a lamb at school

Hey diddle diddle the cat and the fiddle
The cow jumped over the moon
The little dog laughed to see such sport
And the dish ran away with the spoon

Baa baa black sheep have you any wool
Yes sir yes sir three bags full
One for the master and one for the dame
And one for the little boy who lives down the lane

Little Bo Peep has lost her sheep
And doesn't know where to find them
Leave them alone and they'll come home
Bringing their tails behind them

Hickory dickory dock the mouse ran up the clock
The clock struck one the mouse ran down
Hickory dickory dock`,

  json: `{"name": "Alice", "age": 30, "city": "London"}
{"name": "Bob", "age": 25, "city": "Paris"}
{"name": "Charlie", "age": 35, "city": "Berlin"}
{"name": "Diana", "age": 28, "city": "Tokyo"}
{"name": "Edward", "age": 42, "city": "London"}
{"name": "Fiona", "age": 31, "city": "Paris"}
{"name": "George", "age": 27, "city": "Berlin"}
{"name": "Helen", "age": 33, "city": "Tokyo"}
{"name": "Ivan", "age": 29, "city": "London"}
{"name": "Julia", "age": 36, "city": "Paris"}
{"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}
{"users": [{"id": 3, "name": "Charlie"}, {"id": 4, "name": "Diana"}]}
{"config": {"debug": true, "port": 8080, "host": "localhost"}}
{"config": {"debug": false, "port": 3000, "host": "0.0.0.0"}}
{"event": "click", "target": "button", "timestamp": 1234567890}
{"event": "scroll", "target": "page", "timestamp": 1234567891}
{"event": "click", "target": "link", "timestamp": 1234567892}
{"status": "ok", "code": 200, "data": {"items": [1, 2, 3]}}
{"status": "error", "code": 404, "data": {"message": "not found"}}
{"status": "ok", "code": 200, "data": {"items": [4, 5, 6]}}`,
};

export const PRESET_NAMES: { key: string; label: string }[] = [
  { key: 'shakespeare', label: 'Shakespeare' },
  { key: 'rhymes', label: 'Nursery Rhymes' },
  { key: 'json', label: 'JSON' },
];
