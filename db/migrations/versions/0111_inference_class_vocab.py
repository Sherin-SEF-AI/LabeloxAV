"""Which ontology classes a model was able to emit at all, recorded on the run that used it.

The first real shadow sweep compared a 12-class champion against a 9-class challenger and produced 3,091
"challenger missed this" rows out of 3,240. Almost none of them were disagreements. A model that was never
trained on `pedestrian` cannot predict one, so every pedestrian the champion found read as a miss by the
challenger, and a measurement meant to say which model is better instead said which model has more
classes. The same artefact would silently distort every win share computed from those rows.

The fix needs to know what each model could have said, which is a property of the checkpoint rather than
of the corpus, and is knowable only at inference time when the weights are loaded. `class_vocab` is the
list of ontology class ids the model's own class order maps onto, written once per run.

Nullable on purpose, and null is not an empty list: a run recorded before this column existed did not
declare a vocabulary, and the matcher must be able to tell that from a model that genuinely emits nothing.
Faced with null it compares every class, which is exactly what it did before this column existed.

Revision ID: 0111_inference_class_vocab
Revises: 0110_model_lineage
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0111_inference_class_vocab"
down_revision = "0110_model_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("inference_run",
                  sa.Column("class_vocab", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("inference_run", "class_vocab")
