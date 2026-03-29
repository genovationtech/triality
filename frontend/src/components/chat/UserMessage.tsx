interface Props {
  text: string;
}

export function UserMessage({ text }: Props) {
  return (
    <div className="flex justify-end animate-fade-in">
      <div className="max-w-[80%] rounded-2xl rounded-br-md bg-[#6ee7b7]/10 border border-[#6ee7b7]/15 px-4 py-3">
        <p className="text-sm text-white/75 leading-relaxed">{text}</p>
      </div>
    </div>
  );
}
