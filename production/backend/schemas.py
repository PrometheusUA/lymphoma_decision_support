from marshmallow import Schema, fields


class MessageIDSchema(Schema):
    message = fields.Str(required=True)
    id = fields.Int(required=True)


class FileSchema(Schema):
    file = fields.Raw(type="file", required=True)
